using System;
using System.Collections;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using NativeWebSocket;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using static UnityEngine.UIElements.UxmlAttributeDescription;

public class Connection : MonoBehaviour
{
    WebSocket websocket;

    [SerializeField] private BinPLYMeshLoader meshLoader;
    [SerializeField] private RayInteractorSphereSpawner rayClicker;
    [SerializeField] private MaskBasedSubmeshDetacher meshDetacher;

    // Mask chunk handling
    private Dictionary<int, byte[]> maskChunks = new Dictionary<int, byte[]>();
    private int totalExpectedChunks = 0;
    private bool receivingMaskChunks = false;
    private float lastChunkTime;
    private float chunkTimeout = 30f; // 30 seconds timeout

    // Start is called before the first frame update
    async void Start()
    {
        websocket = new WebSocket("ws://localhost:8766");

        websocket.OnOpen += () =>
        {
            Debug.Log("Connection open!");
        };

        websocket.OnError += (e) =>
        {
            Debug.Log("Error! " + e);
        };

        websocket.OnClose += (e) =>
        {
            Debug.Log("Connection closed!");
        };

        websocket.OnMessage += (bytes) =>
        {

            // Convert message bytes to string
            var jsonMessage = System.Text.Encoding.UTF8.GetString(bytes);
            Debug.Log("Received: " + jsonMessage);

            // Process the JSON message
            ProcessJsonMessage(jsonMessage);
        };

        ControlMessages.OnThumbstickPressed += SendNextScene;
        ControlMessages.OnVertexInteraction += SendPointClick;


        // waiting for messages
        await websocket.Connect();
    }

    void ProcessJsonMessage(string jsonMessage)
    {
        try
        {
            // Parse the JSON message using Newtonsoft.Json
            JObject messageObj = JObject.Parse(jsonMessage);

            // Check if the message has a type field
            if (messageObj["type"] == null)
            {
                Debug.LogError("Message missing 'type' field");
                return;
            }

            string messageType = messageObj["type"].ToString();

            // Handle different message types
            switch (messageType)
            {
                case "load_scene":
                    // Reset everything
                    rayClicker.Reset();
                    meshDetacher.ClearDetachedSubmeshes();
                    HandleLoadSceneMessage(messageObj);
                    break;

                case "segmentation_complete":
                    Debug.Log("Segmentation completed");
                    break;

                case "update_mask_chunk":
                    HandleMaskChunk(messageObj);
                    break;

                case "click_feedback":
                    Debug.Log("Click received");
                    break;

                case "error":
                    Debug.LogWarning($"Server error: {messageObj["message"]}");
                    break;


                default:
                    Debug.LogWarning($"Unknown message type: {messageType}");
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error processing message: {e.Message}");
        }
    }

    private async void SendNextScene(bool isPressed)
    {
        var responseMessage = new
        {
            type = "next_scene",
        };

        // Convert to JSON
        string jsonResponse = JsonConvert.SerializeObject(responseMessage);
        // Send the response if the websocket is open
        if (websocket.State == WebSocketState.Open)
        {
            Debug.Log($"Sending next_scene message");
            await websocket.SendText(jsonResponse);
        }
        else
        {
            Debug.LogWarning("WebSocket is not open, cannot send scene_loaded response");
        }
    }

    private async void SendPointClick(int vertexIndex, int labelIndex)
    {
        string click_type = labelIndex == -1 ? "background" : "object";
        var responseMessage = new
        {
            type = "click",
            click_type = click_type,
            point_index = vertexIndex,
            object_id = labelIndex+1
        };

        // Convert to JSON
        string jsonResponse = JsonConvert.SerializeObject(responseMessage);
        // Send the response if the websocket is open
        if (websocket.State == WebSocketState.Open)
        {
            Debug.Log($"Sending click message");
            await websocket.SendText(jsonResponse);
        }
        else
        {
            Debug.LogWarning("WebSocket is not open, cannot send scene_loaded response");
        }
    }

    private async Task HandleLoadSceneMessage(JObject message)
    {
        try
        {
            string sceneName = message["scene_name"].ToString();
            int pointCount = message["point_count"].ToObject<int>();
            string currentObject = message["current_object"].ToString();
            bool semanticsMode = message["semantics_mode"].ToObject<bool>();

            Debug.Log($"Loading scene: {sceneName}");
            Debug.Log($"Point count: {pointCount}");
            Debug.Log($"Current object: {currentObject}");
            Debug.Log($"Semantics mode: {semanticsMode}");

            // Handle the objects array
            if (message["objects"] != null && message["objects"].Type == JTokenType.Array)
            {
                JArray objectsArray = (JArray)message["objects"];

                Debug.Log($"Scene contains {objectsArray.Count} objects");

                // Process each object in the array
                foreach (JObject obj in objectsArray)
                {
                    // Access object properties based on your structure
                    // This is an example - adjust according to your object structure
                    string objName = obj["name"]?.ToString() ?? "unnamed";
                    int objId = obj["id"]?.ToObject<int>() ?? 0;

                    Debug.Log($"Object: {objName}, ID: {objId}");

                    // Process the object further as needed
                }
            }

            if (meshLoader != null)
            {
                string filePath = $"Assets\\Meshes\\scene_{sceneName}\\scan.ply";
                meshLoader.LoadPLY(filePath);
            }
            else
            {
                Debug.LogError("Mesh loader reference not set!");
            }

            var responseMessage = new
            {
                type = "scene_loaded",
                scene_name = sceneName,
                point_count = pointCount
            };

            // Convert to JSON
            string jsonResponse = JsonConvert.SerializeObject(responseMessage);
            // Send the response if the websocket is open
            if (websocket.State == WebSocketState.Open)
            {
                Debug.Log($"Sending scene_loaded response for {sceneName}");
                await websocket.SendText(jsonResponse);
            }
            else
            {
                Debug.LogWarning("WebSocket is not open, cannot send scene_loaded response");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error handling load_scene message: {e.Message}");
        }
    }

    private async Task HandleMaskChunk(JObject data)
    {
        try
        {
            int chunkIndex = data["chunk_index"].ToObject<int>();
            int totalChunks = data["total_chunks"].ToObject<int>();
            int startIndex = data["start_index"].ToObject<int>();
            int endIndex = data["end_index"].ToObject<int>();
            string encodedData = data["data"].ToString();
            bool isCompressed = data["compressed"].ToObject<bool>();

            // Update last chunk time
            lastChunkTime = Time.time;

            // If this is the first chunk, initialize tracking
            if (chunkIndex == 0)
            {
                maskChunks.Clear();
                totalExpectedChunks = totalChunks;
                receivingMaskChunks = true;
                Debug.Log($"Starting to receive mask in {totalChunks} chunks");
            }

            byte[] chunkData = Convert.FromBase64String(encodedData);

            // Store the chunk
            maskChunks[chunkIndex] = chunkData;

            // Convert bytes to int array based on the data format
            int[] maskData = ConvertBytesToMaskData(chunkData);

            // Send this chunk to listeners
            ControlMessages.SendMaskChunk(startIndex, endIndex, maskData);

            // Log progress
            Debug.Log($"Processed mask chunk {chunkIndex + 1}/{totalChunks} " +
                      $"({startIndex}-{endIndex}, {maskData.Length} values)");

            // Check if we have all chunks
            if (maskChunks.Count == totalExpectedChunks)
            {
                Debug.Log("All mask chunks received");
                ControlMessages.SendMaskProcessingComplete();
                maskChunks.Clear();
                receivingMaskChunks = false;
            }

        }
        catch (Exception e)
        {
            Debug.LogError($"Error handling update_mask_chunk message: {e.Message}");
        }
    }

    private int[] ConvertBytesToMaskData(byte[] bytes)
    {
        // Handle the case where data is sent as np.uint8
        // Each value is a single byte (0-255)
        int[] result = new int[bytes.Length];

        for (int i = 0; i < bytes.Length; i++)
        {
            result[i] = (int)bytes[i];
        }

        return result;
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        websocket.DispatchMessageQueue();
#endif
        // Check for timeout on mask chunks
        if (receivingMaskChunks && Time.time - lastChunkTime > chunkTimeout)
        {
            Debug.LogWarning("Mask chunk reception timed out");
            maskChunks.Clear();
            receivingMaskChunks = false;
        }
    }

    async void SendWebSocketMessage()
    {
        if (websocket.State == WebSocketState.Open)
        {
            // Sending bytes
            await websocket.Send(new byte[] { 10, 20, 30 });

            // Sending plain text
            await websocket.SendText("plain text message");
        }
    }

    private async void OnApplicationQuit()
    {
        await websocket.Close();
    }

}