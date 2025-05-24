import asyncio
import websockets
import json
import numpy as np
import time
import threading
import queue
from typing import Dict
import logging
import base64
import zlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveSegmentationVR:
    """VR Server - syncs colors/labels with geometry on VR headset"""

    def __init__(self, segmentation_model, host="0.0.0.0", port=8765):
        self.model = segmentation_model
        self.host = host
        self.port = port
        self.connected_clients = set()

        # Threading
        self.server_thread = None
        self.loop = None
        self.command_queue = queue.Queue()
        self.running = False

        # Scene state (minimal - no geometry)
        self.curr_scene_name = None
        self.num_points = 0
        self.colors = None
        self.points = None
        self.faces = None
        self.labels = None

        # Interaction state
        self.click_idx = {"0": []}
        self.click_time_idx = {"0": []}
        self.click_positions = {"0": []}
        self.cur_obj_idx = -1
        self.cur_obj_name = None
        self.auto_infer = False
        self.num_clicks = 0
        self.cube_size = 0.02
        self.vis_mode_semantics = True

        # Objects
        self.objects = []
        self.current_object_idx = None

        # Configuration for chunked color updates
        self.max_message_size = (
            800 * 1024
        )  # 800KB max per message (well under 1MB limit)
        self.chunk_size = 50000  # Points per chunk

    def run(
        self,
        scene_name,
        point_object,
        coords,
        coords_qv,
        colors,
        original_colors,
        original_labels,
        original_labels_qv,
        is_point_cloud=True,
        object_names=[],
    ):
        """Start VR server - VR already has scene geometry"""
        print(f"Starting VR server for scene: {scene_name}")
        print(f"Scene has {len(coords)} points")

        self.curr_scene_name = scene_name
        self.num_points = len(coords)
        self.colors = colors.copy()
        self.points = point_object.vertices
        self.faces = point_object.triangles
        self.original_colors = original_colors.copy()

        # Store coordinate data for click processing
        self.coordinates = coords
        self.coordinates_qv = coords_qv

        # Initialize objects
        self.objects = list(object_names)
        if object_names:
            self.cur_obj_name = object_names[0]
            self.current_object_idx = 1

        # Start server
        self.start_server_thread()
        time.sleep(1)

        # Send initial scene info (no geometry)
        self.command_queue.put(("send_scene_info", None))

        print(f"VR server ready on ws://{self.host}:{self.port}")
        print("VR headset should:")
        print(f"1. Load scene: {scene_name}")
        print(f"2. Connect to: ws://YOUR_IP:{self.port}")
        print("3. Colors and interactions will be synced")

        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping VR server...")
            self.stop_server()

    def set_new_scene(
        self,
        scene_name,
        point_object,
        coords,
        coords_qv,
        colors,
        original_colors,
        original_labels,
        original_labels_qv,
        is_point_cloud=True,
        object_names=[],
    ):
        """New scene - tell VR to load it"""
        print(f"Loading new scene: {scene_name}")

        self.curr_scene_name = scene_name
        self.num_points = len(coords)
        self.colors = colors.copy()
        self.original_colors = original_colors.copy()
        self.coordinates = coords
        self.coordinates_qv = coords_qv

        # Reset interaction state
        self.click_idx = {"0": []}
        self.click_time_idx = {"0": []}
        self.click_positions = {"0": []}
        self.num_clicks = 0
        self.objects = list(object_names)
        self.current_object_idx = None

        if object_names:
            self.cur_obj_name = object_names[0]
            self.current_object_idx = 1

        # Tell VR to load new scene
        self.command_queue.put(("send_scene_info", None))

    def update_colors(self, colors):
        """Update colors - main sync operation"""
        self.colors = colors.copy()
        self.command_queue.put(("send_colors", colors))

    def select_object(self, colors=None):
        """Select object"""
        if colors is not None:
            self.colors = colors.copy()
            self.command_queue.put(("send_colors", colors))
        self.command_queue.put(("send_objects", None))

    def quit(self, link=""):
        """Quit"""
        print("VR session completed")
        self._queue_message(
            {
                "type": "session_complete",
                "message": "Segmentation completed",
                "link": link,
            }
        )
        self.stop_server()

    # Server implementation

    def start_server_thread(self):
        """Start server thread"""
        if self.server_thread and self.server_thread.is_alive():
            return

        self.running = True
        self.server_thread = threading.Thread(
            target=self._run_server_thread, daemon=True
        )
        self.server_thread.start()

    def stop_server(self):
        """Stop server"""
        self.running = False
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._stop_server(), self.loop)
        if self.server_thread:
            self.server_thread.join(timeout=5)

    def _run_server_thread(self):
        """Server thread"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._start_websocket_server())
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.loop.close()

    async def _start_websocket_server(self):
        """Start WebSocket server"""

        async def handle_client(websocket, path=None):
            await self._register_client(websocket)
            try:
                await self._handle_messages(websocket)
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                logger.error(f"Client error: {e}")
            finally:
                await self._unregister_client(websocket)

        try:
            # Increase max_size to handle larger messages if needed
            server = await websockets.serve(
                handle_client,
                self.host,
                self.port,
                max_size=2 * 1024 * 1024,  # 2MB max message size
                ping_interval=20,
                ping_timeout=10,
            )
            logger.info(f"VR server listening on {self.host}:{self.port}")

            # Process command queue
            while self.running:
                await self._process_commands()
                await asyncio.sleep(0.1)

            server.close()
            await server.wait_closed()
        except Exception as e:
            logger.error(f"Server start error: {e}")

    async def _stop_server(self):
        """Stop server"""
        self.running = False

    async def _register_client(self, websocket):
        """Register VR client"""
        self.connected_clients.add(websocket)
        print(f"VR client connected! ({len(self.connected_clients)} total)")

        # Send current scene info
        if self.curr_scene_name:
            await self._send_scene_info()
            # if self.colors is not None:
            #     await self._send_colors()

    async def _unregister_client(self, websocket):
        """Unregister client"""
        self.connected_clients.discard(websocket)
        print(f"VR client disconnected ({len(self.connected_clients)} remaining)")

    async def _handle_messages(self, websocket):
        """Handle VR messages"""
        async for message in websocket:
            try:
                data = json.loads(message)
                await self._process_vr_message(data, websocket)
            except Exception as e:
                logger.error(f"Message error: {e}")

    async def _process_vr_message(self, data: Dict, websocket):
        """Process VR messages"""
        msg_type = data.get("type")

        if msg_type == "scene_loaded":
            # VR confirms it loaded the scene
            scene_name = data.get("scene_name", "")
            point_count = data.get("point_count", 0)
            print(f"VR confirmed scene loaded: {scene_name} ({point_count} points)")

            # Send initial colors
            # if self.colors is not None:
            #     await self._send_colors()

        elif msg_type == "click":
            await self._handle_vr_click(data)
        elif msg_type == "run_segmentation":
            self.command_queue.put(("run_segmentation", None))
        elif msg_type == "next_scene":
            self.command_queue.put(("next_scene", None))
        elif msg_type == "previous_scene":
            self.command_queue.put(("previous_scene", None))
        elif msg_type == "toggle_colors":
            await self._handle_toggle_colors()
        elif msg_type == "create_object":
            await self._handle_create_object(data.get("name", ""))
        elif msg_type == "switch_object":
            await self._handle_switch_object(data.get("name", ""))
        elif msg_type == "set_auto_infer":
            self.auto_infer = data.get("enabled", False)
        elif msg_type == "heartbeat":
            await self._send_message({"type": "heartbeat_response"}, websocket)
        else:
            print(f"Unknown VR message: {msg_type}")

    async def _process_commands(self):
        """Process command queue"""
        try:
            while not self.command_queue.empty():
                command, data = self.command_queue.get_nowait()

                if command == "send_scene_info":
                    await self._send_scene_info()
                elif command == "send_colors":
                    await self._send_colors(data)
                elif command == "send_objects":
                    await self._send_objects()
                elif command == "run_segmentation":
                    threading.Thread(
                        target=self._run_segmentation_sync, daemon=True
                    ).start()
                elif command == "next_scene":
                    threading.Thread(target=self._next_scene_sync, daemon=True).start()
                elif command == "previous_scene":
                    threading.Thread(
                        target=self._previous_scene_sync, daemon=True
                    ).start()
        except queue.Empty:
            pass

    def _run_segmentation_sync(self):
        """Run segmentation"""
        try:
            if self.vis_mode_semantics and self.num_clicks > 0:
                print(f"Running segmentation with {self.num_clicks} clicks...")
                print(f"Click indices: {self.click_idx}")

                # Ensure all required keys exist and have consistent data
                for key in self.click_idx.keys():
                    if key not in self.click_time_idx:
                        self.click_time_idx[key] = []
                    if key not in self.click_positions:
                        self.click_positions[key] = []

                self.model.get_next_click(
                    click_idx=self.click_idx,
                    click_time_idx=self.click_time_idx,
                    click_positions=self.click_positions,
                    num_clicks=self.num_clicks,
                    run_model=True,
                    gt_labels=None,
                    ori_coords=self.coordinates,
                    scene_name=self.curr_scene_name,
                )
                self._queue_message(
                    {
                        "type": "segmentation_complete",
                        "message": "Segmentation completed!",
                    }
                )
                print("Segmentation completed!")
            else:
                error_msg = f"Cannot run segmentation: semantics_mode={self.vis_mode_semantics}, clicks={self.num_clicks}"
                print(error_msg)
                self._queue_message({"type": "error", "message": error_msg})
        except Exception as e:
            import traceback

            print(f"Segmentation error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            self._queue_message(
                {"type": "error", "message": f"Segmentation error: {str(e)}"}
            )

    def _next_scene_sync(self):
        """Next scene"""
        try:
            self.model.load_next_scene(quit=False)
        except Exception as e:
            print(f"Next scene error: {e}")

    def _previous_scene_sync(self):
        """Previous scene"""
        try:
            self.model.load_next_scene(quit=False, previous=True)
        except Exception as e:
            print(f"Previous scene error: {e}")

    async def _handle_vr_click(self, data: Dict):
        """Handle VR click"""
        if not self.vis_mode_semantics:
            await self._send_message(
                {"type": "error", "message": "Switch to semantic mode first!"}
            )
            return

        click_type = data.get("click_type", "object")
        point_index = data.get("point_index")  # VR sends exact point index
        position = data.get("position", [0, 0, 0])  # For fallback
        obj_id = data.get("object_id", 1)

        print(f"VR {click_type} click - Point index: {point_index}")

        # Use point index if provided, otherwise find nearest
        if point_index is not None:
            point_idx = point_index
            click_position = position
        else:
            # Fallback to position-based lookup
            point_idx = self._find_nearest_point(self.coordinates_qv, position)
            click_position = self.coordinates[
                self._find_nearest_point(self.coordinates, position)
            ].tolist()

        if click_type == "background":
            if "0" not in self.click_idx:
                self.click_idx["0"] = []
                self.click_time_idx["0"] = []
                self.click_positions["0"] = []

            self.click_idx["0"].append(point_idx)
            self.click_time_idx["0"].append(self.num_clicks)
            self.click_positions["0"].append(click_position)

        elif click_type == "object":
            self.cur_obj_idx = obj_id
            obj_key = str(self.cur_obj_idx)

            # Initialize click arrays for this object if they don't exist
            if obj_key not in self.click_idx:
                await self._create_object_internal()
                self.click_idx[obj_key] = []
                self.click_time_idx[obj_key] = []
                self.click_positions[obj_key] = []

            self.click_idx[obj_key].append(point_idx)
            self.click_time_idx[obj_key].append(self.num_clicks)
            self.click_positions[obj_key].append(click_position)

        self.num_clicks += 1
        print(f"Total clicks: {self.num_clicks}")
        print(f"Click data keys: {list(self.click_idx.keys())}")

        # Send click feedback to VR (just the point index and color)
        await self._send_click_feedback(point_idx, click_type, obj_id)

        if self.auto_infer:
            self.command_queue.put(("run_segmentation", None))

    def _find_nearest_point(self, coords, target):
        """Find nearest point (fallback)"""
        if hasattr(coords, "cpu"):
            coords = coords.cpu().numpy()
        else:
            coords = np.array(coords)

        distances = np.sum((coords - np.array(target)) ** 2, axis=1)
        return int(np.argmin(distances))

    async def _handle_toggle_colors(self):
        """Toggle colors"""
        self.vis_mode_semantics = not self.vis_mode_semantics
        colors = self.colors if self.vis_mode_semantics else self.original_colors
        await self._send_colors(colors)
        print(
            f"Toggled to {'semantic' if self.vis_mode_semantics else 'original'} colors"
        )

    async def _handle_create_object(self, name=""):
        """Create object"""
        if not name:
            name = f"object_{len(self.objects) + 1}"

        if name in self.objects:
            await self._send_message({"type": "error", "message": "Object exists!"})
            return

        await self._create_object_internal(name)
        print(f"Created object: {name}")

    async def _create_object_internal(self, name=""):
        """Create object internal"""
        if not name:
            name = f"object_{len(self.objects) + 1}"

        self.objects.append(name)
        self.current_object_idx = len(self.objects)
        self.cur_obj_idx = self.current_object_idx
        self.cur_obj_name = name

        threading.Thread(
            target=lambda: self.model.load_object(name, load_colors=False), daemon=True
        ).start()
        await self._send_objects()

    async def _handle_switch_object(self, name):
        """Switch object"""
        if name not in self.objects:
            await self._send_message({"type": "error", "message": "Object not found!"})
            return

        self.current_object_idx = self.objects.index(name) + 1
        self.cur_obj_idx = self.current_object_idx
        self.cur_obj_name = name

        threading.Thread(
            target=lambda: self.model.load_object(name, load_colors=False), daemon=True
        ).start()
        print(f"Switched to object: {name}")

    # WebSocket communication

    def _queue_message(self, message: Dict):
        """Queue message for VR"""
        if self.loop and self.connected_clients:
            asyncio.run_coroutine_threadsafe(self._send_message(message), self.loop)

    async def _send_message(self, message: Dict, websocket=None):
        """Send message to VR"""
        if websocket:
            try:
                await websocket.send(json.dumps(message))
            except:
                pass
        else:
            if self.connected_clients:
                message_str = json.dumps(message)
                # Check message size
                if len(message_str.encode("utf-8")) > self.max_message_size:
                    logger.warning(f"Message too large: {len(message_str)} bytes")
                    return

                for client in self.connected_clients.copy():
                    try:
                        await client.send(message_str)
                    except:
                        self.connected_clients.discard(client)

    async def _send_scene_info(self):
        """Tell VR which scene to load"""
        message = {
            "type": "load_scene",
            "scene_name": self.curr_scene_name,
            "point_count": self.num_points,
            "objects": self.objects,
            "current_object": self.cur_obj_name,
            "semantics_mode": self.vis_mode_semantics,
        }

        await self._send_message(message)
        print(f"Sent scene info: {self.curr_scene_name}")

    async def _send_colors(self, colors=None):
        """Send color array"""
        if colors is None:
            colors = self.colors

        try:
            # Send colors in chunks to avoid WebSocket size limits
            await self._send_colors_chunked(colors)
            print("Sent color update (chunked)")
        except Exception as e:
            logger.error(f"Color update error: {e}")

    async def _send_colors_chunked(self, colors):
        """chunked to avoid message limits"""
        if hasattr(colors, "cpu"):
            colors = colors.cpu().numpy()

        colors = np.array(colors, dtype=np.float32)  # Use float32 for efficiency
        total_points = len(colors)

        # Calculate number of chunks needed
        num_chunks = (total_points + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_points)

            chunk_colors = colors[start_idx:end_idx]

            # Compress the chunk
            # compressed_data = zlib.compress(chunk_colors.tobytes())
            # encoded_data = base64.b64encode(compressed_data).decode("utf-8")
            encoded_data = base64.b64encode(chunk_colors.tobytes()).decode("utf-8")

            message = {
                "type": "update_colors_chunk",
                "chunk_index": chunk_idx,
                "total_chunks": num_chunks,
                "start_index": start_idx,
                "end_index": end_idx,
                "data": encoded_data,
                "shape": chunk_colors.shape,
                "dtype": str(chunk_colors.dtype),
                "semantics_mode": self.vis_mode_semantics,
                "compressed": True,
            }

            await self._send_message(message)

            # Small delay between chunks to avoid overwhelming the client
            await asyncio.sleep(0.01)

    async def _send_click_feedback(self, point_index, click_type, obj_id):
        """Send immediate click feedback"""
        try:
            from interactive_tool.utils import get_obj_color, BACKGROUND_CLICK_COLOR

            if click_type == "background":
                color = BACKGROUND_CLICK_COLOR
            else:
                color = get_obj_color(obj_id, normalize=True)
        except ImportError:
            # Fallback colors
            if click_type == "background":
                color = [0, 0, 1]  # Blue
            else:
                colors = [[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
                color = colors[obj_id % len(colors)]

        message = {
            "type": "click_feedback",
            "point_index": point_index,
            "color": color,
            "click_type": click_type,
            "object_id": obj_id,
            "total_clicks": self.num_clicks,
        }

        await self._send_message(message)

    async def _send_objects(self):
        """Send objects update"""
        message = {
            "type": "objects_update",
            "objects": self.objects,
            "current_object": self.cur_obj_name,
            "current_object_idx": self.current_object_idx,
        }

        await self._send_message(message)

