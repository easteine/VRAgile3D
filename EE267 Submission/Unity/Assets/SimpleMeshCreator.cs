using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class SimpleMeshCreator : MonoBehaviour
{
    void Start()
    {
        CreateMesh();
    }

    void CreateMesh()
    {
        MeshFilter meshFilter = GetComponent<MeshFilter>();

        Mesh mesh = new Mesh();
        mesh.name = "Simple Quad";

        // Define the vertices of the mesh
        Vector3[] vertices = new Vector3[]
        {
            new Vector3(0, 0, 0),
            new Vector3(1, 0, 0),
            new Vector3(0, 1, 0),
            new Vector3(1, 1, 0)
        };

        // Define the triangles that make up the mesh
        int[] triangles = new int[]
        {
            0, 2, 1, // Lower-left triangle
            2, 3, 1  // Upper-right triangle
        };

        // Define the UV coordinates
        Vector2[] uvs = new Vector2[]
        {
            new Vector2(0, 0),
            new Vector2(1, 0),
            new Vector2(0, 1),
            new Vector2(1, 1)
        };

        // Define the colors for each vertex
        Color[] colors = new Color[]
        {
            Color.red,
            Color.green,
            Color.blue,
            Color.yellow
        };

        // Assign the vertices, triangles, and UV coordinates to the mesh
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.uv = uvs;
        mesh.colors = colors;

        // Recalculate normals and bounds
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        // Assign the mesh to the MeshFilter component
        meshFilter.mesh = mesh;
    }
}