import asyncio
import websockets
import json
import base64
import zlib
import numpy as np

class VRTestClient:
    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.scene_name = None
        self.point_count = 0
        self.color_chunks = {}  # For reassembling chunked color data
        
    async def connect_and_test(self):
        """Connect and run test sequence"""
        try:
            print(f"Connecting to {self.server_url}...")
            # Increase max_size to match server
            self.websocket = await websockets.connect(
                self.server_url,
                max_size=2 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=10
            )
            print("✓ Connected to VR server!")
            
            # Start listening for messages
            listen_task = asyncio.create_task(self.listen_for_messages())
            
            # Run test sequence
            await self.run_test_sequence()
            
            # Cancel listening task
            listen_task.cancel()
            
        except Exception as e:
            print(f"✗ Connection failed: {e}")
        finally:
            if self.websocket:
                await self.websocket.close()
                print("✓ Disconnected")
    
    async def listen_for_messages(self):
        """Listen for server messages"""
        try:
            async for message in self.websocket:
                await self.handle_message(json.loads(message))
        except websockets.exceptions.ConnectionClosed:
            print("✗ Server disconnected")
        except Exception as e:
            print(f"✗ Listen error: {e}")
    
    async def handle_message(self, data):
        """Handle server messages"""
        msg_type = data.get('type', 'unknown')
        
        if msg_type == 'load_scene':
            scene_name = data.get('scene_name', 'Unknown')
            point_count = data.get('point_count', 0)
            objects = data.get('objects', [])
            
            print(f"← Server requests: Load scene '{scene_name}' ({point_count} points)")
            print(f"   Available objects: {objects}")
            
            # Simulate loading scene locally
            self.scene_name = scene_name
            self.point_count = point_count
            
            # Confirm scene loaded
            await self.send_message({
                'type': 'scene_loaded',
                'scene_name': scene_name,
                'point_count': point_count
            })
            print(f"→ Confirmed scene loaded: {scene_name}")
            
        elif msg_type == 'update_colors':
            # Legacy single-message color update
            colors_data = data.get('colors', {})
            shape = colors_data.get('shape', [0, 0])
            semantics_mode = data.get('semantics_mode', True)
            
            num_points = shape[0] if len(shape) > 0 else 0
            mode_str = "semantic" if semantics_mode else "original"
            print(f"← Color update: {num_points} points ({mode_str} mode)")
            
        elif msg_type == 'update_colors_chunk':
            # Handle chunked color updates
            await self.handle_color_chunk(data)
            
        elif msg_type == 'click_feedback':
            point_index = data.get('point_index', -1)
            color = data.get('color', [0, 0, 0])
            click_type = data.get('click_type', 'object')
            total_clicks = data.get('total_clicks', 0)
            
            print(f"← Click feedback: Point {point_index} → {click_type} (color: {color}) [Total: {total_clicks}]")
            
        elif msg_type == 'objects_update':
            objects = data.get('objects', [])
            current = data.get('current_object', 'None')
            print(f"← Objects: {objects}, Current: {current}")
            
        elif msg_type == 'segmentation_complete':
            message = data.get('message', 'Completed')
            print(f"← Segmentation: {message}")
            
        elif msg_type == 'error':
            error_msg = data.get('message', 'Unknown error')
            print(f"← Error: {error_msg}")
            
        elif msg_type == 'session_complete':
            complete_msg = data.get('message', 'Session ended')
            print(f"← Session complete: {complete_msg}")
            
        elif msg_type == 'heartbeat_response':
            print("← Heartbeat OK")
            
        else:
            print(f"← Unknown message: {msg_type}")
    
    async def handle_color_chunk(self, data):
        """Handle chunked color data"""
        chunk_index = data.get('chunk_index', 0)
        total_chunks = data.get('total_chunks', 1)
        start_index = data.get('start_index', 0)
        end_index = data.get('end_index', 0)
        encoded_data = data.get('data', '')
        shape = data.get('shape', [0, 0])
        dtype = data.get('dtype', 'float32')
        semantics_mode = data.get('semantics_mode', True)
        compressed = data.get('compressed', False)
        
        try:
            # Decode the chunk
            if compressed:
                compressed_data = base64.b64decode(encoded_data.encode('utf-8'))
                decompressed_data = zlib.decompress(compressed_data)
                chunk_colors = np.frombuffer(decompressed_data, dtype=dtype).reshape(shape)
            else:
                raw_data = base64.b64decode(encoded_data.encode('utf-8'))
                chunk_colors = np.frombuffer(raw_data, dtype=dtype).reshape(shape)
            
            print(f"← Color chunk {chunk_index + 1}/{total_chunks}: {len(chunk_colors)} points [{start_index}:{end_index}]")
            
            # Store chunk (in a real VR client, you'd update the actual point cloud colors here)
            session_id = f"{self.scene_name}_{semantics_mode}"
            if session_id not in self.color_chunks:
                self.color_chunks[session_id] = {}
            
            self.color_chunks[session_id][chunk_index] = {
                'colors': chunk_colors,
                'start_index': start_index,
                'end_index': end_index
            }
            
            # Check if we have all chunks
            if len(self.color_chunks[session_id]) == total_chunks:
                # All chunks received - could reconstruct full color array here
                total_colors = sum(chunk['end_index'] - chunk['start_index'] 
                                 for chunk in self.color_chunks[session_id].values())
                mode_str = "semantic" if semantics_mode else "original"
                print(f"✓ Complete color update received: {total_colors} points ({mode_str} mode)")
                
                # Clear chunks for this session
                del self.color_chunks[session_id]
        
        except Exception as e:
            print(f"✗ Error processing color chunk: {e}")
    
    async def send_message(self, message):
        """Send message to server"""
        if self.websocket:
            try:
                await self.websocket.send(json.dumps(message))
            except Exception as e:
                print(f"✗ Send error: {e}")
    
    async def run_test_sequence(self):
        """Run complete test sequence"""
        print("\n" + "="*60)
        print("VR SERVER TEST")
        print("="*60)
        
        # Wait for server to send scene info
        print("1. Waiting for scene load request...")
        await asyncio.sleep(3)
        
        if not self.scene_name:
            print("   No scene load request received - continuing anyway")
        
        # Test heartbeat
        print("2. Testing heartbeat...")
        await self.send_message({"type": "heartbeat"})
        await asyncio.sleep(1)
        
        # Test object creation
        print("3. Creating test object...")
        await self.send_message({
            "type": "create_object",
            "name": "test_object_1"
        })
        await asyncio.sleep(2)  # Give more time for color updates
        
        # Test point-based clicking (new approach)
        print("4. Testing object click with point index...")
        await self.send_message({
            "type": "click",
            "click_type": "object",
            "point_index": 37206,  # Specific point index
            "position": [0.1, 0.1, 0.1],  # Fallback position
            "object_id": 1
        })
        await asyncio.sleep(1)
        
        # Test background click
        print("5. Testing background click...")
        await self.send_message({
            "type": "click", 
            "click_type": "background",
            "point_index": 2000,
            "position": [-0.1, -0.1, -0.1]
        })
        await asyncio.sleep(1)
        
        # Test more object clicks
        print("6. Adding more object clicks...")
        for i in range(3):
            await self.send_message({
                "type": "click",
                "click_type": "object", 
                "point_index": 1500 + i * 100,
                "position": [0.2 + i * 0.1, 0.2, 0.1],
                "object_id": 1
            })
            await asyncio.sleep(0.5)
        
        # Test segmentation
        print("7. Running segmentation...")
        await self.send_message({"type": "run_segmentation"})
        await asyncio.sleep(8)  # Give more time for segmentation and color updates
        
        # Test color toggle
        print("8. Toggling colors...")
        await self.send_message({"type": "toggle_colors"})
        await asyncio.sleep(3)  # Give time for chunked color update
        
        # Test auto-infer
        print("9. Enabling auto-infer...")
        await self.send_message({
            "type": "set_auto_infer",
            "enabled": True
        })
        await asyncio.sleep(1)
        
        # Toggle back to semantic mode for auto-infer test
        print("10. Toggling back to semantic mode...")
        await self.send_message({"type": "toggle_colors"})
        await asyncio.sleep(3)
        
        # Test click with auto-infer
        print("11. Testing click with auto-infer...")
        await self.send_message({
            "type": "click",
            "click_type": "object",
            "point_index": 3000,
            "position": [0.5, 0.5, 0.5],
            "object_id": 1
        })
        await asyncio.sleep(5)  # Auto-segmentation should trigger with color updates
        
        print("12. Test sequence completed!")
        print("="*60)
        
        # Keep connection alive for a bit
        print("Keeping connection alive for 10 seconds...")
        await asyncio.sleep(10)

async def main():
    """Main test function"""
    import sys
    
    server_url = "ws://localhost:8765"
    if len(sys.argv) > 1 and sys.argv[1].startswith("ws://"):
        server_url = sys.argv[1]
    
    print("VR Server Test Client")
    
    client = VRTestClient(server_url)
    await client.connect_and_test()

if __name__ == "__main__":
    print("Usage:")
    print("  python test_client.py                    # Test localhost")
    print("  python test_client.py ws://IP:8765       # Test custom server")
    print()
    
    asyncio.run(main())
