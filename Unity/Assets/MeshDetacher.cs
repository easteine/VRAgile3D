using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR.Interaction.Toolkit;
using UnityEngine.XR.Interaction.Toolkit.Interactables;
using UnityEngine.XR.Interaction.Toolkit.Interactors;

public class MaskBasedSubmeshDetacher : MonoBehaviour
{
    [Header("References")]
    [SerializeField] private RayInteractorSphereSpawner vertexMasker; // Reference to your vertex masking script
    [SerializeField] private XRRayInteractor rayInteractor;
    [SerializeField] private InputActionProperty detachButtonAction; // Button to trigger detachment

    [Header("Detached Mesh Settings")]
    [SerializeField] private Material detachedMeshMaterial;
    [SerializeField] private float meshOffset = 0.0f; // How far to offset detached meshes
    [SerializeField] private bool makeInteractable = true;
    [SerializeField] private bool useOriginalColors = true;

    [Header("Physics")]
    [SerializeField] private float mass = 1f;
    [SerializeField] private bool useGravity = false;

    private bool wasDetachPressed = false;
    private List<GameObject> detachedSubmeshes = new List<GameObject>();

    private void Awake()
    {
        // Find references if not set
        if (vertexMasker == null)
        {
            vertexMasker = FindObjectOfType<RayInteractorSphereSpawner>();
            if (vertexMasker == null)
            {
                Debug.LogError("Cannot find RayInteractorSphereSpawner script. Please assign it in the Inspector.");
            }
        }

        if (rayInteractor == null)
        {
            rayInteractor = GetComponent<XRRayInteractor>();
        }

        if (detachedMeshMaterial == null)
        {
            // Create a default material
            detachedMeshMaterial = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        }
    }

    private void OnEnable()
    {
        if (detachButtonAction.action != null)
        {
            detachButtonAction.action.Enable();
        }
    }

    private void OnDisable()
    {
        if (detachButtonAction.action != null)
        {
            detachButtonAction.action.Disable();
        }
    }

    private void Update()
    {
        if (detachButtonAction.action != null)
        {
            bool isDetachPressed = detachButtonAction.action.IsPressed();

            if (isDetachPressed && !wasDetachPressed)
            {
                DetachSubmeshes();
            }

            wasDetachPressed = isDetachPressed;
        }
    }

    private void DetachSubmeshes()
    {
        if (rayInteractor.TryGetCurrent3DRaycastHit(out RaycastHit hit))
        {
            GameObject hitObject = hit.collider.gameObject;
            MeshFilter meshFilter = hitObject.GetComponent<MeshFilter>();

            if (meshFilter != null && meshFilter.mesh != null)
            {
                Mesh currentMesh = meshFilter.mesh;

                // Get access to vertex masks from the vertex masker script
                int[] vertexMask = vertexMasker.GetVertexMask(hitObject);

                if (vertexMask == null || vertexMask.Length == 0)
                {
                    Debug.Log("No vertex mask found for this object. Has it been colored?");
                    return;
                }

                // Get the original mesh if needed
                Mesh originalMesh = vertexMasker.GetOriginalMesh(hitObject);
                Mesh sourceMesh = originalMesh != null ? originalMesh : currentMesh;

                // Group vertices by mask index
                Dictionary<int, List<int>> maskGroups = new Dictionary<int, List<int>>();

                for (int i = 0; i < vertexMask.Length; i++)
                {
                    int maskValue = vertexMask[i];

                    // Skip unlabeled vertices
                    if (maskValue == -1) continue;

                    if (!maskGroups.ContainsKey(maskValue))
                    {
                        maskGroups[maskValue] = new List<int>();
                    }

                    maskGroups[maskValue].Add(i);
                }

                if (maskGroups.Count == 0)
                {
                    Debug.Log("No labeled vertices found in the mask.");
                    return;
                }

                // Get the color options for reference
                List<RayInteractorSphereSpawner.ColorOption> colorOptions = vertexMasker.GetColorOptions();
                // Track all vertices to be removed from original
                HashSet<int> verticesToRemove = new HashSet<int>();

                // Create submeshes for each mask group
                foreach (var group in maskGroups)
                {
                    int maskIndex = group.Key;
                    List<int> vertexIndices = group.Value;

                    Color groupColor = (maskIndex >= 0 && maskIndex < colorOptions.Count)
                        ? colorOptions[maskIndex].color
                        : Color.gray;

                    string groupName = (maskIndex >= 0 && maskIndex < colorOptions.Count)
                        ? colorOptions[maskIndex].name
                        : "Unknown";

                    CreateSubmeshFromVertices(hitObject, sourceMesh, vertexIndices, groupColor, groupName);

                    foreach (int idx in vertexIndices)
                    {
                        verticesToRemove.Add(idx);
                    }
                }

                // Remove detached parts from original mesh and set
                if (verticesToRemove.Count > 0)
                {
                    RemoveVerticesFromMesh(hitObject, sourceMesh, verticesToRemove);
                }
                


                Debug.Log($"Created {maskGroups.Count} submeshes from mask groups");
            }
            else
            {
                Debug.LogWarning("Hit object doesn't have a MeshFilter or Mesh");
            }
        }
    }

    private void RemoveVerticesFromMesh(GameObject obj, Mesh originalMesh, HashSet<int> verticesToRemove)
    {

        // Get the original data
        Vector3[] origVertices = originalMesh.vertices;
        Vector3[] origNormals = originalMesh.normals;
        Vector2[] origUVs = originalMesh.uv;
        Color[] origColors = originalMesh.colors;
        int[] origTriangles = originalMesh.triangles;

        //// Build a map of old->new vertex indices
        //Dictionary<int, int> oldToNewVertexMap = new Dictionary<int, int>();
        //List<Vector3> newVertices = new List<Vector3>();
        //List<Vector3> newNormals = new List<Vector3>();
        //List<Vector2> newUVs = new List<Vector2>();
        //List<Color> newColors = new List<Color>();

        // Create new triangles array without removed vertices
        List<int> newTriangles = new List<int>();

        for (int i = 0; i < origTriangles.Length; i += 3)
        {
            int v1 = origTriangles[i];
            int v2 = origTriangles[i + 1];
            int v3 = origTriangles[i + 2];

            // Only keep triangles that don't use any removed vertices
            if (!verticesToRemove.Contains(v1) ||
                !verticesToRemove.Contains(v2) ||
                !verticesToRemove.Contains(v3))
            {
                // Map old indices to new ones
                newTriangles.Add(v1);
                newTriangles.Add(v2);
                newTriangles.Add(v3);
            }
        }

        // Create a new mesh without the removed parts
        //Mesh newMesh = new Mesh();
        Mesh newMesh = Instantiate(originalMesh);
        newMesh.name = originalMesh.name + "_Reduced";

        // Set the data
        //newMesh.SetVertices(newVertices);
        //if (newNormals.Count > 0) newMesh.SetNormals(newNormals);
        //if (newUVs.Count > 0) newMesh.SetUVs(0, newUVs);
        //if (newColors.Count > 0) newMesh.SetColors(newColors);
        newMesh.SetTriangles(newTriangles, 0);

        // Recalculate bounds and normals if needed
        newMesh.RecalculateBounds();
        //if (newNormals.Count == 0) newMesh.RecalculateNormals();

        // Update the mesh filter
        //meshFilter.mesh = newMesh;

        // Update the mesh collider if it exists
        MeshCollider meshCollider = obj.GetComponent<MeshCollider>();
        if (meshCollider != null)
        {
            meshCollider.sharedMesh = newMesh;
        }

        // Notify the vertex masker that the mesh has changed
        //vertexMasker.NotifyMeshChanged(obj, newMesh);
        vertexMasker.SetNewMesh(obj, newMesh);
    }

    private void CreateSubmeshFromVertices(GameObject sourceObj, Mesh sourceMesh, List<int> vertexIndices, Color color, string groupName)
    {
        if (vertexIndices.Count == 0) return;

        // Original data
        Vector3[] originalVertices = sourceMesh.vertices;
        Vector3[] originalNormals = sourceMesh.normals;
        Vector2[] originalUVs = sourceMesh.uv;
        Color[] originalColors = sourceMesh.colors;
        int[] originalTriangles = sourceMesh.triangles;

        // Build a set of the vertex indices for fast lookup
        HashSet<int> vertexIndexSet = new HashSet<int>(vertexIndices);

        // Find triangles that only use vertices from our set
        List<int> submeshTriangles = new List<int>();
        Dictionary<int, int> oldToNewVertexMap = new Dictionary<int, int>();

        for (int i = 0; i < originalTriangles.Length; i += 3)
        {
            // Only include triangles where all 3 vertices are in our set
            if (vertexIndexSet.Contains(originalTriangles[i]) &&
                vertexIndexSet.Contains(originalTriangles[i + 1]) &&
                vertexIndexSet.Contains(originalTriangles[i + 2]))
            {
                // For each vertex in the triangle
                for (int j = 0; j < 3; j++)
                {
                    int oldIndex = originalTriangles[i + j];

                    // Map old vertex index to new vertex index
                    if (!oldToNewVertexMap.ContainsKey(oldIndex))
                    {
                        oldToNewVertexMap[oldIndex] = oldToNewVertexMap.Count;
                    }

                    // Add to the new triangle list with the mapped index
                    submeshTriangles.Add(oldToNewVertexMap[oldIndex]);
                }
            }
        }

        if (submeshTriangles.Count == 0)
        {
            Debug.Log($"No complete triangles found for mask group {groupName}");
            return;
        }

        // Create new mesh with only the selected vertices
        Mesh submesh = new Mesh();
        submesh.name = $"Submesh_{groupName}";

        // Create new vertex arrays
        Vector3[] newVertices = new Vector3[oldToNewVertexMap.Count];
        Vector3[] newNormals = originalNormals.Length > 0 ? new Vector3[oldToNewVertexMap.Count] : null;
        Vector2[] newUVs = originalUVs.Length > 0 ? new Vector2[oldToNewVertexMap.Count] : null;
        Color[] newColors = originalColors.Length > 0 ? new Color[oldToNewVertexMap.Count] : null;

        // Map data from old indices to new
        foreach (var mapping in oldToNewVertexMap)
        {
            int oldIndex = mapping.Key;
            int newIndex = mapping.Value;

            newVertices[newIndex] = originalVertices[oldIndex];

            if (newNormals != null)
                newNormals[newIndex] = originalNormals[oldIndex];

            if (newUVs != null)
                newUVs[newIndex] = originalUVs[oldIndex];

            if (newColors != null && useOriginalColors)
                newColors[newIndex] = originalColors[oldIndex];
            else if (newColors != null)
                newColors[newIndex] = color;
        }

        // Apply to new mesh
        submesh.vertices = newVertices;
        if (newNormals != null) submesh.normals = newNormals;
        if (newUVs != null) submesh.uv = newUVs;
        if (newColors != null) submesh.colors = newColors;
        submesh.triangles = submeshTriangles.ToArray();

        // Recalculate if needed
        submesh.RecalculateBounds();
        if (newNormals == null) submesh.RecalculateNormals();

        // Create a new GameObject for the submesh
        GameObject submeshObj = new GameObject($"Detached_{groupName}");
        submeshObj.transform.position = sourceObj.transform.position + (sourceObj.transform.up * meshOffset);
        submeshObj.transform.rotation = sourceObj.transform.rotation;
        submeshObj.transform.localScale = sourceObj.transform.localScale;

        // Add components
        MeshFilter newMeshFilter = submeshObj.AddComponent<MeshFilter>();
        MeshRenderer newRenderer = submeshObj.AddComponent<MeshRenderer>();
        MeshCollider newCollider = submeshObj.AddComponent<MeshCollider>();

        // Assign mesh
        newMeshFilter.mesh = submesh;
        newCollider.sharedMesh = submesh;
        newCollider.convex = true;

        // Assign material
        Material instanceMaterial = new Material(detachedMeshMaterial);
        instanceMaterial.color = color;
        newRenderer.material = instanceMaterial;

        // Make interactable if requested
        if (makeInteractable)
        {
            Rigidbody rb = submeshObj.AddComponent<Rigidbody>();
            rb.mass = mass;
            rb.useGravity = useGravity;
            rb.linearDamping = 100f;         // Set linear damping
            rb.angularDamping = 100f;  // Set angular damping

            XRGrabInteractable grabInteractable = submeshObj.AddComponent<XRGrabInteractable>();
            grabInteractable.movementType = XRBaseInteractable.MovementType.VelocityTracking;
            grabInteractable.throwOnDetach = true;
        }

        // Track the detached submesh
        detachedSubmeshes.Add(submeshObj);
    }

    // Method to destroy all detached submeshes
    public void ClearDetachedSubmeshes()
    {
        foreach (var submesh in detachedSubmeshes)
        {
            if (submesh != null)
            {
                Destroy(submesh);
            }
        }

        detachedSubmeshes.Clear();
    }

    private void OnDestroy()
    {
        ClearDetachedSubmeshes();
    }
}