from plyfile import PlyData

plydata = PlyData.read('scan.ply')

plydata.text = True

plydata.write('scan_ascii.ply')
