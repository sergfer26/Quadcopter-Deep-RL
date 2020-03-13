import bpy 
import numpy as np 

X = np.loadtxt('/Users/papiit/Desktop/Reinforcement_Learning/Proyecto2/XYZ.txt')
A = np.loadtxt('/Users/papiit/Desktop/Reinforcement_Learning/Proyecto2/ang.txt')

posiciones, angulos = [],[]
for i in range(len(X[0])):
    posiciones.append([X[0][i],X[1][1],X[2][i]])
    angulos.append([A[0][i],A[1][1],A[2][i]])
ob = bpy.data.objects["drone"]
frame_num = 0
for i in range(len(angulos)):
    bpy.context.scene.frame_set(frame_num)
    ob.location = posiciones[i]
    ob.rotation_euler = angulos[i]
    ob.keyframe_insert(data_path = "location",index = -1)
    ob.keyframe_insert("rotation_euler", frame= frame_num)
    frame_num += 1
