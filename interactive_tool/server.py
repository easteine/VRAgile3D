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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveSegmentationGUI:
    """VR Server that replaces the original GUI with the same interface"""
    
    def __init__(self, segmentation_model, host='0.0.0.0', port=8765):
        self.model = segmentation_model
        self.host = host
        self.port = port
        self.websocket = None
        self.connected_clients = set()
        
        # Threading components
        self.server_thread = None
        self.loop = None
        self.command_queue = queue.Queue()
        self.running = False
        
        # Scene state (same as original GUI)
        self.curr_scene_name = None
        self.original_colors = None
        self.coordinates = None
        self.coordinates_qv = None
        self.points = None
        self.is_point_cloud = True
        self.old_colors = None
        self.new_colors = None
        self.original_labels = None
        self.original_labels_qv = None
        
        # Interaction state
        self.click_idx = {'0': []}
        self.click_time_idx = {'0': []}
        self.click_positions = {'0': []}
        self.cur_obj_idx = -1
        self.cur_obj_name = None
        self.auto_infer = False
        self.num_clicks = 0
        self.cube_size = 0.02
        self.vis_mode_semantics = True
        
        # Object management
        self.objects = []
        self.current_object_idx = None
        
        # VR state
        self.point_size = 2.0
    
    def run(self, scene_name, point_object, coords, coords_qv, colors, original_colors,
            original_labels, original_labels_qv, is_point_cloud=True, object_names=[]):
        """Start VR server - same interface as original GUI"""
        print(f"Starting VR server for scene: {scene_name}")
        print(f"Point cloud has {len(coords)} points")
        
        self.curr_scene_name = scene_name
        self.init_points(point_object, coords, coords_qv, colors, original_colors, 
                        original_labels, original_labels_qv, is_point_cloud)
        
        # Initialize objects
        self.objects = []
        if object_names:
            for obj in object_names:
                self.objects.append(obj)
                self.current_object_idx = len(self.objects)
                self.cur_obj_idx = self.current_object_idx
                self.cur_obj_name = obj
                self.model.load_object(obj, load_colors=False)
        
        # Start server
        self.start_server_thread()
        time.sleep(1)  # Wait for server to start
        
        # Queue initial scene data
        self.command_queue.put(('send_scene_update', None))
        
        print(f"VR server ready on ws://{self.host}:{self.port}")
        print("Connect your VR headset to start segmentation!")
        
        # Keep running
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping VR server...")
            self.stop_server()
    
    def set_new_scene(self, scene_name, point_object, coords, coords_qv, colors,
                     original_colors, original_labels, original_labels_qv,
                     is_point_cloud=True, object_names=[]):
        """Set new scene - same interface as original GUI"""
        print(f"Loading new scene: {scene_name}")
        
        self.curr_scene_name = scene_name
        self.init_points(point_object, coords, coords_qv, colors, original_colors,
                        original_labels, original_labels_qv, is_point_cloud)
        
        # Reset state
        self.click_idx = {'0': []}
        self.click_time_idx = {'0': []}
        self.num_clicks = 0
        self.objects = []
        self.current_object_idx = None
        
        if object_names:
            for obj in object_names:
                self.objects.append(obj)
                self.current_object_idx = len(self.objects)
                self.cur_obj_idx = self.current_object_idx
                self.cur_obj_name = obj
                self.model.load_object(obj, load_colors=False)
        
        # Send to VR
        self.command_queue.put(('send_scene_update', None))
    
    def update_colors(self, colors):
        """Update colors - same interface as original GUI"""
        self.old_colors = colors.copy()
        self.new_colors = colors.copy()
        self.command_queue.put(('send_color_update', None))
    
    def select_object(self, colors=None):
        """Select object - same interface as original GUI"""
        if colors is not None:
            self.old_colors = colors.copy()
            self.new_colors = colors.copy()
            self.command_queue.put(('send_color_update', None))
        self.command_queue.put(('send_objects_update', None))
    
    def quit(self, link=""):
        """Quit - same interface as original GUI"""
        print("VR session completed")
        self._queue_message({
            'type': 'quit',
            'message': 'Segmentation completed',
            'link': link
        })
        self.stop_server()
    
    # Server implementation
    
    def start_server_thread(self):
        """Start WebSocket server in background thread"""
        if self.server_thread and self.server_thread.is_alive():
            return
        
        self.running = True
        self.server_thread = threading.Thread(target=self._run_server_thread, daemon=True)
        self.server_thread.start()
    
    def stop_server(self):
        """Stop the server"""
        self.running = False
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._stop_server(), self.loop)
        if self.server_thread:
            self.server_thread.join(timeout=5)
    
    def _run_server_thread(self):
        """Run server in separate thread"""
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
                logger.error(f"Client handler error: {e}")
            finally:
                await self._unregister_client(websocket)
        
        try:
            server = await websockets.serve(handle_client, self.host, self.port)
            logger.info(f"WebSocket server started on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return
        
        # Process commands
        while self.running:
            await self._process_commands()
            await asyncio.sleep(0.1)
        
        server.close()
        await server.wait_closed()
    
    async def _stop_server(self):
        """Stop server gracefully"""
        self.running = False
    
    async def _register_client(self, websocket):
        """Register VR client"""
        self.connected_clients.add(websocket)
        self.websocket = websocket
        print(f"VR client connected! ({len(self.connected_clients)} total)")
        
        if self.curr_scene_name:
            await self._send_scene_update()
    
    async def _unregister_client(self, websocket):
        """Unregister VR client"""
        self.connected_clients.discard(websocket)
        if self.websocket == websocket:
            self.websocket = None
        print(f"VR client disconnected ({len(self.connected_clients)} remaining)")
    
    async def _handle_messages(self, websocket):
        """Handle VR messages"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._process_vr_message(data, websocket)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from client: {e}")
                    await self._send_message({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }, websocket)
                except Exception as e:
                    logger.error(f"Message processing error: {e}")
                    await self._send_message({
                        'type': 'error', 
                        'message': f'Processing error: {str(e)}'
                    }, websocket)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed normally")
        except Exception as e:
            logger.error(f"Message handler error: {e}")
    
    async def _process_vr_message(self, data: Dict, websocket):
        """Process VR commands"""
        msg_type = data.get('type')
        
        if msg_type == 'click':
            await self._handle_vr_click(data)
        elif msg_type == 'run_segmentation':
            self.command_queue.put(('run_segmentation', None))
        elif msg_type == 'next_scene':
            self.command_queue.put(('next_scene', None))
        elif msg_type == 'previous_scene':
            self.command_queue.put(('previous_scene', None))
        elif msg_type == 'toggle_colors':
            await self._handle_toggle_colors()
        elif msg_type == 'create_object':
            await self._handle_create_object(data.get('name', ''))
        elif msg_type == 'switch_object':
            await self._handle_switch_object(data.get('name', ''))
        elif msg_type == 'set_auto_infer':
            self.auto_infer = data.get('enabled', False)
        elif msg_type == 'heartbeat':
            await self._send_message({'type': 'heartbeat_response'}, websocket)
    
    async def _process_commands(self):
        """Process queued commands"""
        try:
            while not self.command_queue.empty():
                command, data = self.command_queue.get_nowait()
                
                if command == 'run_segmentation':
                    threading.Thread(target=self._run_segmentation_sync, daemon=True).start()
                elif command == 'next_scene':
                    threading.Thread(target=self._next_scene_sync, daemon=True).start()
                elif command == 'previous_scene':
                    threading.Thread(target=self._previous_scene_sync, daemon=True).start()
                elif command == 'send_scene_update':
                    await self._send_scene_update()
                elif command == 'send_color_update':
                    await self._send_color_update()
                elif command == 'send_objects_update':
                    await self._send_objects_update()
        except queue.Empty:
            pass
    
    def _run_segmentation_sync(self):
        """Run segmentation (sync)"""
        try:
            if self.vis_mode_semantics and self.num_clicks > 0:
                print(f"Running segmentation with {self.num_clicks} clicks...")
                self.model.get_next_click(
                    click_idx=self.click_idx,
                    click_time_idx=self.click_time_idx,
                    click_positions=self.click_positions,
                    num_clicks=self.num_clicks,
                    run_model=True,
                    gt_labels=getattr(self, 'new_labels', None),
                    ori_coords=self.coordinates,
                    scene_name=self.curr_scene_name
                )
                self._queue_message({
                    'type': 'segmentation_complete',
                    'message': 'Segmentation completed!'
                })
                print("Segmentation completed successfully!")
        except Exception as e:
            print(f"Segmentation error: {e}")
            self._queue_message({
                'type': 'error',
                'message': f'Error: {str(e)}'
            })
    
    def _next_scene_sync(self):
        """Load next scene (sync)"""
        try:
            self.model.load_next_scene(quit=False)
        except Exception as e:
            print(f"Next scene error: {e}")
    
    def _previous_scene_sync(self):
        """Load previous scene (sync)"""
        try:
            self.model.load_next_scene(quit=False, previous=True)
        except Exception as e:
            print(f"Previous scene error: {e}")
    
    async def _handle_vr_click(self, data: Dict):
        """Handle VR click"""
        if not self.vis_mode_semantics:
            await self._send_message({
                'type': 'error',
                'message': 'Toggle to semantic mode first!'
            })
            return
        
        click_type = data.get('click_type', 'object')
        position = data.get('position', [0, 0, 0])
        obj_id = data.get('object_id', 1)
        
        print(f"VR {click_type} click at {position}")
        
        # Find nearest point
        point_idx = self._find_nearest_point(self.coordinates_qv, position)
        click_position = self.coordinates[self._find_nearest_point(self.coordinates, position)].tolist()
        
        # Create mask
        mask = self._create_segmentation_mask(position)
        if mask.sum() <= 0:
            return
        
        if click_type == 'background':
            self.click_idx['0'].append(point_idx)
            self.click_time_idx['0'].append(self.num_clicks)
            self.click_positions['0'].append(click_position)
            
            # Import colors
            try:
                from interactive_tool.utils import BACKGROUND_CLICK_COLOR
                self.new_colors[mask] = BACKGROUND_CLICK_COLOR
            except ImportError:
                self.new_colors[mask] = [0, 0, 1]  # Blue fallback
            
        elif click_type == 'object':
            self.cur_obj_idx = obj_id
            
            if self.click_idx.get(str(self.cur_obj_idx)) is None:
                await self._create_object_internal()
                self.click_idx[str(self.cur_obj_idx)] = [point_idx]
                self.click_time_idx[str(self.cur_obj_idx)] = [self.num_clicks]
                self.click_positions[str(self.cur_obj_idx)] = [click_position]
                
                if hasattr(self, 'new_labels') and self.new_labels is not None:
                    self.new_labels[self.original_labels == self.original_labels_qv[point_idx]] = self.cur_obj_idx
            else:
                self.click_idx[str(self.cur_obj_idx)].append(point_idx)
                self.click_time_idx[str(self.cur_obj_idx)].append(self.num_clicks)
                self.click_positions[str(self.cur_obj_idx)].append(click_position)
            
            # Set object color
            try:
                from interactive_tool.utils import get_obj_color
                self.new_colors[mask] = get_obj_color(self.cur_obj_idx, normalize=True)
            except ImportError:
                # Fallback colors
                colors = [[1,0,0], [0,1,0], [1,1,0], [1,0,1], [0,1,1]]
                self.new_colors[mask] = colors[self.cur_obj_idx % len(colors)]
        
        self.num_clicks += 1
        print(f"Total clicks: {self.num_clicks}")
        
        # Send updates
        await self._send_color_update()
        await self._send_click_info_update()
        
        if self.auto_infer:
            self.command_queue.put(('run_segmentation', None))
    
    def _find_nearest_point(self, coords, target):
        """Find nearest point"""
        if hasattr(coords, 'cpu'):
            coords = coords.cpu().numpy()
        else:
            coords = np.array(coords)
        
        distances = np.sum((coords - np.array(target)) ** 2, axis=1)
        return int(np.argmin(distances))
    
    def _create_segmentation_mask(self, position):
        """Create segmentation mask"""
        mask = np.zeros([self.coordinates.shape[0]], dtype=bool)
        coords = self.coordinates
        
        mask[np.logical_and(np.logical_and(
            (np.abs(coords[:, 0] - position[0]) < self.cube_size),
            (np.abs(coords[:, 1] - position[1]) < self.cube_size)), 
            (np.abs(coords[:, 2] - position[2]) < self.cube_size))] = True
        
        return mask
    
    async def _handle_toggle_colors(self):
        """Toggle colors"""
        self.vis_mode_semantics = not self.vis_mode_semantics
        colors = self.new_colors if self.vis_mode_semantics else self.original_colors
        await self._send_color_update(colors)
        print(f"Toggled to {'semantic' if self.vis_mode_semantics else 'original'} colors")
    
    async def _handle_create_object(self, name=""):
        """Create object"""
        if not name:
            name = f'object {len(self.objects) + 1}'
        
        if name in self.objects:
            await self._send_message({'type': 'error', 'message': 'Object exists!'})
            return
        
        await self._create_object_internal(name)
        print(f"Created object: {name}")
    
    async def _create_object_internal(self, name=""):
        """Create object internal"""
        if not name:
            name = f'object {len(self.objects) + 1}'
        
        self.objects.append(name)
        self.current_object_idx = len(self.objects)
        self.cur_obj_idx = self.current_object_idx
        self.cur_obj_name = name
        
        threading.Thread(target=lambda: self.model.load_object(name, load_colors=False), daemon=True).start()
        await self._send_objects_update()
    
    async def _handle_switch_object(self, name):
        """Switch object"""
        if not self.vis_mode_semantics:
            await self._send_message({'type': 'error', 'message': 'Toggle to semantic mode first!'})
            return
        
        if name not in self.objects:
            await self._send_message({'type': 'error', 'message': 'Object not found!'})
            return
        
        self.current_object_idx = self.objects.index(name) + 1
        self.cur_obj_idx = self.current_object_idx
        self.cur_obj_name = name
        
        threading.Thread(target=lambda: self.model.load_object(name, load_colors=False), daemon=True).start()
        print(f"Switched to object: {name}")
    
    def init_points(self, point_object, coords, coords_qv, colors, original_colors,
                   original_labels, original_labels_qv, is_point_cloud):
        """Initialize point data"""
        self.vis_mode_semantics = True
        self.coordinates = coords
        self.coordinates_qv = coords_qv
        self.old_colors = colors.copy()
        self.new_colors = colors.copy()
        self.original_colors = original_colors
        self.original_labels = original_labels
        self.original_labels_qv = original_labels_qv
        self.is_point_cloud = is_point_cloud
        self.points = point_object
        
        if self.original_labels is not None:
            import torch
            self.new_labels = torch.zeros(self.original_labels.shape, 
                                        device=self.original_labels.device)
        else:
            self.new_labels = None
    
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
                for client in self.connected_clients.copy():
                    try:
                        await client.send(message_str)
                    except:
                        self.connected_clients.discard(client)
    
    async def _send_scene_update(self):
        """Send scene to VR"""
        if not hasattr(self, 'coordinates') or self.coordinates is None:
            return
        
        coords_data = self._encode_array(self.coordinates)
        colors_data = self._encode_array(self.new_colors)
        
        message = {
            'type': 'scene_update',
            'scene_name': self.curr_scene_name,
            'coordinates': coords_data,
            'colors': colors_data,
            'is_point_cloud': self.is_point_cloud,
            'point_size': self.point_size,
            'objects': self.objects,
            'current_object': self.cur_obj_name,
            'semantics_mode': self.vis_mode_semantics
        }
        
        await self._send_message(message)
        print(f"Sent scene update: {len(self.coordinates)} points")
    
    async def _send_color_update(self, colors=None):
        """Send color update to VR"""
        if colors is None:
            colors = self.new_colors
        
        colors_data = self._encode_array(colors)
        
        message = {
            'type': 'color_update',
            'colors': colors_data,
            'semantics_mode': self.vis_mode_semantics
        }
        
        await self._send_message(message)
    
    async def _send_objects_update(self):
        """Send objects update"""
        message = {
            'type': 'objects_update',
            'objects': self.objects,
            'current_object': self.cur_obj_name,
            'current_object_idx': self.current_object_idx
        }
        
        await self._send_message(message)
    
    async def _send_click_info_update(self):
        """Send click info update"""
        message = {
            'type': 'click_info_update',
            'num_clicks': self.num_clicks,
            'auto_infer': self.auto_infer
        }
        
        await self._send_message(message)
    
    def _encode_array(self, array):
        """Encode array for transmission"""
        if hasattr(array, 'cpu'):
            array = array.cpu().numpy()
        
        # Convert to float32 for efficiency
        array = np.array(array, dtype=np.float32)
        
        return {
            'data': base64.b64encode(array.tobytes()).decode('utf-8'),
            'shape': array.shape,
            'dtype': str(array.dtype)
        }