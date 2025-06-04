using System.Collections.Generic;
using System.IO;
using UnityEngine;

[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class PLYMeshLoader : MonoBehaviour
{
    void Start()
    {
        LoadPLY("/home/codeysun/My project/Assets/scan_ascii.ply");
    }

    void LoadPLY(string path)
    {
        if (!File.Exists(path))
        {
            Debug.LogError("File not found: " + path);
            return;
        }

        List<Vector3> vertices = new List<Vector3>();
        List<Color> colors = new List<Color>();
        List<int> triangles = new List<int>();

        using (StreamReader sr = new StreamReader(path))
        {
            string line;
            bool header = true;
            bool readingVertices = false;
            bool readingFaces = false;
            int vertexCount = 0;
            int faceCount = 0;
            int vertexRead = 0;
            int faceRead = 0;

            while ((line = sr.ReadLine()) != null)
            {
                if (header)
                {
                    if (line.StartsWith("element vertex"))
                    {
                        vertexCount = int.Parse(line.Split(' ')[2]);
                    }
                    else if (line.StartsWith("element face"))
                    {
                        faceCount = int.Parse(line.Split(' ')[2]);
                    }
                    else if (line.StartsWith("end_header"))
                    {
                        header = false;
                        readingVertices = true;
                    }
                }
                else if (readingVertices)
                {
                    // x y z r g b a
                    if (vertexRead < vertexCount)
                    {
                        string[] parts = line.Split(' ');
                        Vector3 vertex = new Vector3(
                            float.Parse(parts[0]),
                            float.Parse(parts[1]),
                            float.Parse(parts[2])
                        );

                        Color color = new Color(
                            int.Parse(parts[3]) / 255f,
                            int.Parse(parts[4]) / 255f,
                            int.Parse(parts[5]) / 255f,
                            int.Parse(parts[6]) / 255f
                        );

                        vertices.Add(vertex);
                        colors.Add(color);

                        vertexRead++;
                    }

                    if (vertexRead == vertexCount)
                    {
                        readingVertices = false;
                        readingFaces = true;
                    }
                }
                else if (readingFaces)
                {
                    if (faceRead < faceCount)
                    {
                        string[] parts = line.Split(' ');
                        int faceVertexCount = int.Parse(parts[0]);

                        for (int i = 1; i < faceVertexCount - 1; i++)
                        {
                            triangles.Add(int.Parse(parts[1]));
                            triangles.Add(int.Parse(parts[i + 1]));
                            triangles.Add(int.Parse(parts[i + 2]));
                        }

                        faceRead++;
                    }
                }
            }
        }

        Mesh mesh = new Mesh();
        mesh.name = "PLY Mesh";
        mesh.SetVertices(vertices);
        mesh.SetTriangles(triangles, 0);
        mesh.SetColors(colors);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        MeshFilter meshFilter = GetComponent<MeshFilter>();
        meshFilter.mesh = mesh;
    }
}