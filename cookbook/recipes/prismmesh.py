"""
Make and plot a 3D prism mesh.
"""
import fatiando as ft

log = ft.log.get()
log.info(ft.log.header())
log.info(__doc__)

mesh = ft.msh.ddd.PrismMesh(bounds=(-2, 2, -3, 3, 0, 1), shape=(4,4,4))

ft.vis.figure3d()
plot = ft.vis.prisms(mesh)
axes = ft.vis.axes3d(plot)
ft.vis.show3d()
