Shader "Custom/VertexColorShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" { }
    }
    SubShader
    {
        Tags { "Queue" = "Geometry" }
        Lighting Off
        ZWrite On
        ZTest LEqual
        Cull Back
        Fog { Mode Off }
        // Blend SrcAlpha OneMinusSrcAlpha // Allow transparency, if needed

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float4 color : COLOR;
                UNITY_VERTEX_INPUT_INSTANCE_ID //Insert
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float4 color : COLOR;
                UNITY_VERTEX_OUTPUT_STEREO //Insert
            };

            v2f vert(appdata v)
            {
                v2f o;

                UNITY_SETUP_INSTANCE_ID(v); //Insert
                UNITY_INITIALIZE_OUTPUT(v2f, o); //Insert
                UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO(o); //Insert

                o.pos = UnityObjectToClipPos(v.vertex);
                o.color = v.color;
                return o;
            }

            UNITY_DECLARE_SCREENSPACE_TEXTURE(_MainTex); //Insert

            half4 frag(v2f i) : SV_Target
            {
                UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(i); //Insert
    
                return i.color;
            }
            ENDCG
        }
    }
    
    FallBack "Diffuse"
}