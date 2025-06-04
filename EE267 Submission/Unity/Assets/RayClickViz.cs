using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactors;
using static UnityEngine.Mesh;

public class RayInteractorSphereSpawner : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private XRRayInteractor rayInteractor;
    [SerializeField] private InputActionProperty triggerAction;

    [Header("Button Controls for Color Selection")]
    [SerializeField] private InputActionProperty primaryButtonAction; // Usually 'A' or 'X' button
    [SerializeField] private InputActionProperty secondaryButtonAction; // Usually 'B' or 'Y' button
    [SerializeField] private InputActionProperty thumbstickAction;

    [Header("Vertex Coloring")]
    [SerializeField] private float maxVertexSearchDistance = 0.5f;
    [SerializeField] private float vertexColorRadius = 0.5f; // For multi-vertex coloring
    [SerializeField] private bool colorMultipleVertices = true;
    [SerializeField] private bool resetColorsOnNewSelection = false;

    [Header("Color Options")]
    [SerializeField] private List<ColorOption> colorOptions = new List<ColorOption>();
    [SerializeField] private int currentColorIndex = 0;
    Color backgroundColor = Color.white;

    [System.Serializable]
    public class ColorOption
    {
        public string name = "Color";
        public Color color = Color.red;
    }

    private bool wasPressed = false;
    private bool wasPrimaryPressed = false;
    //private bool wasSecondaryPressed = false;
    private bool wasThumbstickPressed = false;

    //private Dictionary<Mesh, Mesh> originalMeshDict = new Dictionary<Mesh, Mesh>();
    //private Dictionary<Mesh, Color[]> originalColorDict = new Dictionary<Mesh, Color[]>();
    //private Dictionary<Mesh, GameObject> currentlyColoredMeshes = new Dictionary<Mesh, GameObject>();
    //// Store vertex color index masks (-1 = unlabeled, otherwise = index in colorOptions)
    //private Dictionary<Mesh, int[]> vertexColorIndexMasks = new Dictionary<Mesh, int[]>();

    private Dictionary<GameObject, Mesh> originalMeshDict = new Dictionary<GameObject, Mesh>();
    private Dictionary<GameObject, Color[]> originalColorDict = new Dictionary<GameObject, Color[]>();
    private Dictionary<GameObject, int[]> vertexColorIndexMasks = new Dictionary<GameObject, int[]>();
    private HashSet<GameObject> currentlyColoredObjects = new HashSet<GameObject>();
    private Mesh currentMesh = null;

    public void Reset()
    {
        currentColorIndex = 0;
        originalMeshDict.Clear();
        originalColorDict.Clear();
        vertexColorIndexMasks.Clear();
        currentlyColoredObjects.Clear();
        currentMesh = null;
    }
    private void Awake()
    {
        // If no ray interactor assigned, try to get it from this GameObject
        if (rayInteractor == null)
        {
            rayInteractor = GetComponent<XRRayInteractor>();
        }

        // Initialize with default colors if none are defined
        if (colorOptions.Count == 0)
        {
            colorOptions.Add(new ColorOption { name = "Red", color = Color.red });
            colorOptions.Add(new ColorOption { name = "Green", color = Color.green });
            colorOptions.Add(new ColorOption { name = "Blue", color = Color.blue });
            colorOptions.Add(new ColorOption { name = "Yellow", color = Color.yellow });
            colorOptions.Add(new ColorOption { name = "Cyan", color = Color.cyan });
            //colorOptions.Add(new ColorOption { name = "Magenta", color = Color.magenta });
            //colorOptions.Add(new ColorOption { name = "White", color = Color.white });
            //colorOptions.Add(new ColorOption { name = "Black", color = Color.black });
        }
    }

    private void OnEnable()
    {
        triggerAction.action.Enable();

        primaryButtonAction.action.Enable();
        secondaryButtonAction.action.Enable();
        thumbstickAction.action.Enable();

        ControlMessages.OnMaskChunkReceived += ColorMeshMaskChunk;
        ControlMessages.OnMaskProcessingComplete += OnMaskProcessingComplete;
    }

    private void OnDisable()
    {
        triggerAction.action.Disable();

        primaryButtonAction.action.Disable();
        secondaryButtonAction.action.Disable();
        thumbstickAction.action.Disable();
        ControlMessages.OnMaskChunkReceived -= ColorMeshMaskChunk;
        ControlMessages.OnMaskProcessingComplete -= OnMaskProcessingComplete;
    }

    private void Update()
    {
        // Handle color selection
        HandleColorSelection();

        // Handle next/prev scene
        HandleSceneSelection();

        // Handle vertex coloring
        bool isPressed = triggerAction.action.ReadValue<float>() > 0.5f;
        if (isPressed && !wasPressed)
        {
            ColorClosestVertexAtRaycastHit();
        }
        wasPressed = isPressed;
    }

    private void HandleSceneSelection()
    {
        if (thumbstickAction.action != null)
        {
            bool isThumbstickPressed = thumbstickAction.action.ReadValue<float>() > 0.5f;
            if (isThumbstickPressed && !wasThumbstickPressed)
            {
                ControlMessages.SendThumbstickPressed(isThumbstickPressed);
                Debug.Log($"Next scene clicked");
            }
            wasThumbstickPressed = isThumbstickPressed;
        }
    }


    private void HandleColorSelection()
    {
        // Primary button (typically A or X) - next color
        if (primaryButtonAction.action != null)
        {
            bool isPrimaryPressed = primaryButtonAction.action.ReadValue<float>() > 0.5f;
            if (isPrimaryPressed && !wasPrimaryPressed)
            {
                currentColorIndex = (currentColorIndex + 1) % colorOptions.Count;
                Debug.Log($"Color changed to: {colorOptions[currentColorIndex].name}");
            }
            wasPrimaryPressed = isPrimaryPressed;
        }

    }

    private void ColorMeshMaskChunk(int startIndex, int endIndex, int[] maskChunkData)
    {
        if (currentlyColoredObjects.Count == 0)
        {
            Debug.LogWarning("No mesh to color");
            return;
        }
        GameObject hitObject = currentlyColoredObjects.First();
        MeshFilter meshFilter = hitObject.GetComponent<MeshFilter>();
        Mesh mesh = meshFilter.mesh;

        // TODO: update vertexColorIndexMasks
        int chunkOffset = 0;
        for (int i = startIndex; i < endIndex && i < vertexColorIndexMasks[hitObject].Length; i++)
        {
            if (chunkOffset < maskChunkData.Length)
            {
                int labelValue = maskChunkData[chunkOffset++];
                vertexColorIndexMasks[hitObject][i] = labelValue-1;
            }
        }

        Debug.Log("Mask chunk received");
    }

    // Called when all chunks have been processed
    private void OnMaskProcessingComplete()
    {
        Debug.Log("All mask chunks have been processed");

        // Additional finalization if needed
        if (currentlyColoredObjects.Count > 0)
        {
            GameObject hitObject = currentlyColoredObjects.First();
            MeshFilter meshFilter = hitObject.GetComponent<MeshFilter>();
            Mesh mesh = meshFilter.mesh;
            Mesh originalMesh = originalMeshDict[hitObject];

            int[] vertexMask = vertexColorIndexMasks[hitObject];

            // Color the mesh
            Color[] colors = originalMesh.colors;

            for (int i = 0; i < colors.Length; i++)
            {
                if (vertexMask[i] >= 0)
                {
                    colors[i] = colorOptions[vertexMask[i]].color;
                }                
            }

            mesh.colors = colors;
        }

        Debug.Log($"AGILE3D mask received and colored");
    }
    private void ColorClosestVertexAtRaycastHit()
    {
        // Check if ray hits anything
        if (rayInteractor.TryGetCurrent3DRaycastHit(out RaycastHit hit))
        {
            // Get the mesh from the hit object
            //MeshFilter meshFilter = hit.collider.GetComponent<MeshFilter>();
            GameObject hitObject = hit.collider.gameObject;
            MeshFilter meshFilter = hitObject.GetComponent<MeshFilter>();
            if (meshFilter != null && meshFilter.mesh != null)
            {
                Mesh mesh = meshFilter.mesh;
                Transform hitTransform = hit.collider.transform;

                // If this is our first time interacting with this object
                if (!currentlyColoredObjects.Contains(hitObject))
                {
                    // Store the original for later restoration
                    originalMeshDict[hitObject] = mesh;


                    // Create a working copy we'll modify
                    Mesh workingCopy = Instantiate(mesh);

                    // Replace the mesh filter's mesh with our working copy
                    meshFilter.mesh = workingCopy;

                    // Update our reference to the mesh we'll be modifying
                    mesh = workingCopy;

                    // Initialize colors if needed
                    if (mesh.colors == null || mesh.colors.Length == 0)
                    {
                        Color[] initcolors = new Color[mesh.vertexCount];
                        for (int i = 0; i < initcolors.Length; i++)
                        {
                            initcolors[i] = Color.white;
                        }
                        mesh.colors = initcolors;
                    }

                    // Initialize the vertex mask
                    int[] initvertexMask = new int[mesh.vertexCount];
                    for (int i = 0; i < initvertexMask.Length; i++)
                    {
                        initvertexMask[i] = -1; // -1 indicates unlabeled
                    }
                    vertexColorIndexMasks[hitObject] = initvertexMask;

                    // Track this mesh as currently being colored
                    currentlyColoredObjects.Add(hitObject);
                }

                // Reset colors on previously colored meshes if needed
                if (resetColorsOnNewSelection && currentlyColoredObjects.Count > 0)
                {
                    RestoreAllOriginalColors();
                    // Restart the coloring process
                    ColorClosestVertexAtRaycastHit();
                    return;
                }

                Vector3[] vertices = mesh.vertices;
                Color[] colors = mesh.colors;
                int[] vertexMask = vertexColorIndexMasks[hitObject];

                // Find the closest vertex
                float closestDistance = float.MaxValue;
                int closestVertexIndex = -1;

                for (int i = 0; i < vertices.Length; i++)
                {
                    // Convert local vertex position to world space
                    Vector3 worldVertex = hitTransform.TransformPoint(vertices[i]);

                    // Calculate distance to hit point
                    float distance = Vector3.Distance(worldVertex, hit.point);

                    // If this vertex is closer than the previous closest
                    if (distance < closestDistance && distance <= maxVertexSearchDistance)
                    {
                        closestDistance = distance;
                        closestVertexIndex = i;
                    }
                }

                // Get the current selected color
                Color selectedColor = colorOptions[currentColorIndex].color;
                string colorName = colorOptions[currentColorIndex].name;
                int labelIndex = currentColorIndex;

                if (secondaryButtonAction.action.ReadValue<float>() > 0.5f)
                {
                    selectedColor = backgroundColor;
                    colorName = "background";
                    labelIndex = -1;
                }

                if (closestVertexIndex >= 0)
                {
                    // Send click info to websocket
                    ControlMessages.SendVertexInteraction(
                        closestVertexIndex,
                        labelIndex
                    );

                    if (colorMultipleVertices)
                    {
                        // Color the closest vertex and nearby vertices based on radius
                        Vector3 closestVertexPos = vertices[closestVertexIndex];

                        for (int i = 0; i < vertices.Length; i++)
                        {
                            float localDistance = Vector3.Distance(vertices[i], closestVertexPos);
                            if (localDistance <= vertexColorRadius)
                            {
                                colors[i] = selectedColor;
                                vertexMask[i] = labelIndex;
                            }
                        }
                    }
                    else
                    {
                        // Just color the single closest vertex
                        colors[closestVertexIndex] = selectedColor;
                        vertexMask[closestVertexIndex] = labelIndex;
                    }

                    // Apply the updated colors
                    mesh.colors = colors;

                    Debug.Log($"Vertex colored with {colorName} at index: {closestVertexIndex}");
                }
                else
                {
                    Debug.Log("No vertices found within max search distance");
                }
            }
            else
            {
                Debug.LogWarning("Hit object doesn't have a MeshFilter or Mesh");
            }
        }
    }

    private void RestoreOriginalColors(GameObject obj)
    {
        if (originalMeshDict.TryGetValue(obj, out Mesh originalMesh))
        {
            // Get the mesh filter
            MeshFilter meshFilter = obj.GetComponent<MeshFilter>();
            if (meshFilter != null)
            {
                // Restore the original mesh
                meshFilter.mesh = originalMesh;

                // Clean up dictionaries
                originalMeshDict.Remove(obj);
                originalColorDict.Remove(obj);
                vertexColorIndexMasks.Remove(obj);
                currentlyColoredObjects.Remove(obj);
            }
        }
    }

    private void RestoreAllOriginalColors()
    {
        // Create a copy of the list to avoid modification during iteration
        List<GameObject> objectsToRestore = new List<GameObject>(currentlyColoredObjects);

        foreach (GameObject obj in objectsToRestore)
        {
            RestoreOriginalColors(obj);
        }

        // Make sure all collections are cleared
        originalMeshDict.Clear();
        originalColorDict.Clear();
        vertexColorIndexMasks.Clear();
        currentlyColoredObjects.Clear();
    }

    private void OnDestroy()
    {
        // Clean up and restore colors
        RestoreAllOriginalColors();
    }

    // Get the vertex mask for a GameObject
    public int[] GetVertexMask(GameObject obj)
    {
        if (vertexColorIndexMasks.TryGetValue(obj, out int[] mask))
        {
            // Return a copy to prevent external modification
            int[] maskCopy = new int[mask.Length];
            System.Array.Copy(mask, maskCopy, mask.Length);
            return maskCopy;
        }
        return null;
    }

    // Get the original mesh for a GameObject
    public Mesh GetOriginalMesh(GameObject obj)
    {
        if (originalMeshDict.TryGetValue(obj, out Mesh originalMesh))
        {
            return originalMesh;
        }
        return null;
    }

    // Get the color options list
    public List<ColorOption> GetColorOptions()
    {
        return new List<ColorOption>(colorOptions);
    }

    public void SetNewMesh(GameObject obj, Mesh mesh)
    {
        MeshFilter meshFilter = obj.GetComponent<MeshFilter>();
        meshFilter.mesh = mesh;
    }
}