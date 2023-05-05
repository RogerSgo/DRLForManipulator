import sim
import numpy as np
import cv2 as cv
import random
import time

STEREO_CAMERA_NAME = "StereoCamera_"
JOINT_NAME = "LBR4p_joint" # nombre de la articulacion
SCENE_CONTROLLER_NAME = "SceneController"

JOINT_RANGE_FUNC_NAME = "getJointRange" # Funcion en coppelia devuelve las coordenadas minimas y maximas de las articulaciones

RESET_FUNC_NAME = "resetItems" # Funcion en coppelia organiza aleatoriamente los objetos y devuelve el angulo del sector del objeto de destino

JOINT_COUNT = 6 # numero de articulaciones

class Robot():
    """
    Interaccion del manipulador con la escena

    Atributos
    ---------
    client : int
        Id del cliente conectado al entorno CoppeliaSim
    is_connected : bool
        Verificar si el cliente esta conectado con el entorno
    synchronous : bool
        Verificar si esta activado el modo de sincronizacion
    cameras : list
        Id de las camaras que forman una camara estereo
    joints : list
        Id de articulaciones
    joint_ranges : ndarray
        Coordenadas extremas generalizadas
    default_pos : ndarray
        Coordenadas de la posicion de inicio
    scene_controller : int
        Id del objeto que controla el entorno del robot
    stereo_matcher : cv2.StereoBM
        Objeto que calcula el mapa de profundidad

    Metodos
    -------
    get_vision_feedback()
        Obtener retroalimentacion visual
    get_adometry_feedback()
        Obtener retroalimentacion de posicion
    clip_position(coords)
        Llevar las coordenadas a un rango aceptable
    set_target_position(coords)
        Establecer posicion de destino (para escena dinamica)
    set_position(coords)
        Establecer posicion (para escena estatica)
    move(coords_t)
        Mover a una posicion (para escena dinamica)
    reset(is_dynamic=False, do_orientate=True)
        Restablecer escena
    enable_synchronization()
        Habilitar modo de sincronizacion
    disable_synchronization()
        Desactivar el modo de sincronizacion
    """

    def __init__(self):
        sim.simxFinish(-1) # Cerrar todas las conexiones abiertas
        self.client=sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5) # connect
        self.is_connected = self.client != -1
        self.synchronous = False
        self.cameras = []
        self.joints = []
        self.joint_ranges = np.zeros((JOINT_COUNT,2), dtype=np.float32)
        self.default_pos = np.asarray([0, 0, 2.97, 2.62, 1.57, 0])
        self.scene_controller = 0
        self.stereo_matcher = cv.StereoBM_create(numDisparities=48, blockSize=11)

        if self.is_connected:
            _, self.scene_controller = sim.simxGetObjectHandle(self.client, SCENE_CONTROLLER_NAME, sim.simx_opmode_blocking)
            for i in range(1,3):
                _, id = sim.simxGetObjectHandle(self.client, STEREO_CAMERA_NAME + str(i), sim.simx_opmode_blocking)
                self.cameras.append(id)
            for i in range(1,JOINT_COUNT+1):
                _, id = sim.simxGetObjectHandle(self.client, JOINT_NAME + str(i), sim.simx_opmode_blocking)
                self.joints.append(id)
                _, _, min_max, _, _ = sim.simxCallScriptFunction(self.client, SCENE_CONTROLLER_NAME, sim.sim_scripttype_childscript, JOINT_RANGE_FUNC_NAME,
                    [id],[],[],bytearray(), sim.simx_opmode_blocking)
                for j in range(2):
                    self.joint_ranges[i-1,j] = min_max[j]
            self.reset()
        else:
            print('No se pudo conectar al servidor API remoto')

    def __del__(self):
        if self.synchronous:
            self.disable_synchronization()
        sim.simxFinish(self.client)
    
    def get_vision_feedback(self):
        """
            Obtener retroalimentacion visual

            Retorno
            -------
            ndarray
                Imagen de la camara en RGB
            ndarray
                Mapa de profundidad estereo
        """

        # Imagenes stereo
        imgs = []
        for i in range(2):
            _, res, img = sim.simxGetVisionSensorImage(self.client, self.cameras[i], False, sim.simx_opmode_blocking)
            img = np.asarray(img, dtype=np.uint8)
            img = np.reshape(img, (res[0],res[1],3))
            img = np.flip(img, axis=0)
            imgs.append(img)
        
        # Mapa de profundidad
        left = cv.cvtColor(imgs[1], cv.COLOR_RGB2GRAY)
        right = cv.cvtColor(imgs[0], cv.COLOR_RGB2GRAY)
        depth_map = self.stereo_matcher.compute(left,right)
        return imgs[1], depth_map/752

    def get_adometry_feedback(self):
        """
            Obtener retroalimentacion de posicion

            Retorno
            -------
            ndarray
                Coordenadas generalizadas
        """
        pos = np.zeros((JOINT_COUNT))
        for i in range(JOINT_COUNT):
            _, pos[i] = sim.simxGetJointPosition(self.client, self.joints[i], sim.simx_opmode_blocking)
        return self.clip_position(pos)

    def clip_position(self, coords):
        """
            Llevar las coordenadas a un rango aceptable

            Parametros
            ----------
            coords : ndarray
                Coordenadas generalizadas "inseguras"

            Retorno
            -------
            ndarray
                Coordenadas generalizadas "seguras"
        """

        return np.clip(coords, self.joint_ranges[:,0], self.joint_ranges[:,1])

    def set_target_position(self, coords):
        """
            Establecer posicion de destino (para escena dinamica)

            Parametros
            ----------
            coords : ndarray
                Coordenadas generalizadas
        """

        sim.simxPauseCommunication(self.client, 1)
        for i in range(JOINT_COUNT):
            sim.simxSetJointTargetPosition(self.client, self.joints[i], coords[i], sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.client, 0)

    def set_position(self, coords):
        """
            Establecer posicion (para escena estatica)

            No funciona para escena dinamica. Para estatica, provoca un movimiento instantaneo.
            Al trabajar con comentarios, debe usarse en modo de sincronizacion para evitar obtener datos obsoletos.

            Parametros
            ----------
            coords : ndarray
                Coordenadas generalizadas
        """

        sim.simxPauseCommunication(self.client, 1)
        for i in range(JOINT_COUNT):
            sim.simxSetJointPosition(self.client, self.joints[i], coords[i], sim.simx_opmode_oneshot)
        sim.simxPauseCommunication(self.client, 0)
        if self.synchronous:
            self._step_simulation()

    def move(self, coords_t):
        """
            Moverse a una posicion (para una escena dinamica).

            No se admite en el modo de sincronizacion. Establece una posicion de destino y espera a que se alcance 
            a traves de la retroalimentacion. La funcion terminara la ejecucion si no se puede alcanzar el objetivo.

            Parametros
            ----------
            coords_t : ndarray
                Coordenadas generalizadas

            Returns
            -------
            bool
                se ha alcanzado la posicion

            Raises
            ------
            ValueError
                No compatible en modo de sincronizacion
        """

        if self.synchronous:
            raise ValueError()
        self.set_target_position(coords_t)
        last_e = 1000
        while True:
            coords = self.get_adometry_feedback()
            e = np.mean(np.abs(coords - coords_t))
            if e < 0.01:
                return True
            if last_e - e < 0.001:
                return False
            last_e = e
            time.sleep(0.2)

    def reset(self, is_dynamic=False, do_orientate=True):
        """
            Restablecer escena

            Los objetos de escena adquieren una posicion aleatoria en el area de trabajo del manipulador.
            El robot pasa a la configuracion inicial. Si do_orientate=True, entonces el primer eje se dirige 
            hacia el objeto, de lo contrario, toma un valor aleatorio en el rango de trabajo.

            Parametros
            ----------
            is_dynamic : bool, optional
                Es la escena dinamica, por defecto Falso
            do_orientate : bool, opcional
                Si el robot gira hacia el objeto, por defecto True
        """

        # phi - punto del sector en el que se genero el target
        _, _, phi, _, _ = sim.simxCallScriptFunction(self.client, SCENE_CONTROLLER_NAME, sim.sim_scripttype_childscript, RESET_FUNC_NAME,[],[],[],bytearray(), sim.simx_opmode_blocking)
        pos = self.default_pos.copy()
        #print('phi', phi)
        if do_orientate:
            offset = random.random()*0.2 - 0.1 # Pequena desviacion en el campo de vision del objeto
            pos[0] = phi[0] + offset
        else:
            pos[0] = random.random()*(self.joint_ranges[0,1]-self.joint_ranges[0,0]) + self.joint_ranges[0,0]
        
        if is_dynamic:
            self.move(pos)
        else:
            self.set_position(pos)

    def enable_synchronization(self):
        """
            Habilitar el modo de sincronizacion
        """
        sim.simxSynchronous(self.client,True)
        self.synchronous = True

    def disable_synchronization(self):
        """
            Desactivar el modo de sincronizacion
        """
        sim.simxSynchronous(self.client,False)
        self.synchronous =False

    def _step_simulation(self):
        """
            Paso de simulacion para el modo de temporizacion
        """
        sim.simxSynchronousTrigger(self.client)
        sim.simxGetPingTime(self.client)