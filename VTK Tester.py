import vtk
from vtk.util import numpy_support
import numpy as np
from pyevtk.hl import gridToVTK
import matplotlib.pyplot as plt

import SimpleITK as sitk

#replace SetInput to SetInputConnection
#replace GetOutput with GetOutputPort()
"""
cube = vtk.vtkCubeSource()

cube_mapper = vtk.vtkPolyDataMapper()
cube_mapper.SetInputConnection(cube.GetOutputPort())

cube_actor = vtk.vtkActor()
cube_actor.SetMapper(cube_mapper)
cube_actor.GetProperty().SetColor(1.0, 0.0, 0.0)

renderer = vtk.vtkRenderer()
renderer.SetBackground(0.0, 0.0, 0.0)
renderer.AddActor(cube_actor)

render_window = vtk.vtkRenderWindow()
render_window.SetWindowName("Simple VTK scene")
render_window.SetSize(400, 400)
render_window.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

interactor.Initialize()
render_window.Render()
interactor.Start()
"""
def sample_stack(stack, rows=2, cols=2, start_with=0, show_every=1, display1 = True):
    if (display1):
        new_list = []
        new_list.append(stack)
        new_list.append(stack)
        new_list.append(stack)
        new_list.append(stack)
        sample_stack(new_list, 2, 2, 0, 1, False)
    else:
        fig,ax = plt.subplots(rows,cols,figsize=[12,12])
        for i in range((rows*cols)):
            ind = start_with + i*show_every
            ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
            ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
            ax[int(i/rows),int(i % rows)].axis('off')
        plt.show()
id = 0
imgs_after_rg = np.load("/Users/paulmccabe/Desktop/" + "trachea_to_model_%d.npy" % (id))
print(np.amax(imgs_after_rg[150]))
h, r, c = imgs_after_rg.shape
dataImporter = vtk.vtkImageImport()


data_string = imgs_after_rg.tostring()
dataImporter.CopyImportVoidPointer(data_string, len(data_string))

dataImporter.SetDataScalarTypeToUnsignedChar()

dataImporter.SetNumberOfScalarComponents(1)

dataImporter.SetDataExtent(0, h, 0, r, 0, c)
dataImporter.SetWholeExtent(0, h, 0, r, 0, c)
print("alphaChannel")
alphaChannelFunc = vtk.vtkPiecewiseFunction()
alphaChannelFunc.AddPoint(255, 0.6)
print("colorFunc")
colorFunc = vtk.vtkColorTransferFunction()
colorFunc.AddRGBPoint(0, 0.0, 0.0, 0.0)
colorFunc.AddRGBPoint(255, 0.0, 1.0, 0.0)

print("VolumeProperty")
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorFunc)
volumeProperty.SetScalarOpacity(alphaChannelFunc)
print("CompositeFunction")
compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
print("Volume Mapper")
volumeMapper = vtk.vtkVolumeRayCastMapper()
volumeMapper.SetVolumeRayCastFunction(compositeFunction)
volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
print("Volume set mapper")
volume = vtk.vtkVolume()
volume.SetMapper(volumeMapper)
volume.SetProperty(volumeProperty)
print("renderer")
renderer = vtk.vtkRenderer()
renderWin = vtk.vtkRenderWindow()
renderWin.AddRenderer(renderer)
renderInteractor = vtk.vtkRenderWindowInteractor()
renderInteractor.SetRenderWindow(renderWin)
print("add volume")
renderer.AddVolume(volume)

renderer.SetBackground(0,0,0)
renderWin.SetSize(500, 500)

def exitCheck(obj, event):
    if obj.GetEventPending() != 0:
        obj.SetAbortRender(1)

renderWin.AddObserver("AbortCheckEvent", exitCheck)
print("Render")

renderInteractor.Initialize()
renderWin.Render()
renderInteractor.Start()
print("all done")


"""
h, r, c = imgs_after_rg.shape
x = np.arange(0, h+1)
y = np.arange(0, r+1)
z = np.arange(0, c+1)
gridToVTK("./trachea_img", x, y, z, cellData = {'trachea_img': imgs_after_rg})
VTK_data = numpy_support.numpy_to_vtk(num_array=imgs_after_rg.ravel(), deep=True, array_type=vtk.VTK_INT)

filename = "trachea_img.vtr"
reader = vtk.vtkXMLRectilinearGridReader()
reader.SetFileName(filename)
reader.Update()

airway_mapper = vtk.vtkDataSetMapper()
#airway_mapper = vtk.vtkPolyDataMapper()
airway_mapper.SetInputConnection(reader.GetOutputPort())

airway_actor = vtk.vtkActor()
airway_actor.SetMapper(airway_mapper)
airway_actor.GetProperty().SetColor(1.0, 0.0, 0.0)

renderer = vtk.vtkRenderer()
renderer.SetBackground(0.0, 0.0, 0.0)
renderer.AddActor(airway_actor)

renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(renderWindow)

interactor.Initialize()
renderWindow.Render()
interactor.Start()
"""
#save vtk data
#gridToVTK("./julia", x, y, z, cellData = {'julia': imgs_after_rg})
