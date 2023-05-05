import numpy as np
import network_utils

SEARCH_RESOLUTION = 20 

class Searcher():
    """
        Clase responsable de buscar un objeto a lo largo de la primera coordenada.

        Atributos
        -------
        bounds : list or ndarray
            Coordenadas minimas y maximas
        res : int
            Resolucion de busqueda: el numero de posiciones en el rango para verificar
        map : ndarray
            Resultados de verificacion de posicion (-1 - no encontrado, 0 - desconocido, 1 - encontrado)

        Metodos
        -------
        get_coord(current_coord)
            Obtener siguiente coordenada
        mark_coord(coord, is_found)
            Marcar resultado de verificacion por coordenada
        reset()
            Restablecer resultados de la prueba
    """

    def __init__(self, bounds):
        self.bounds = bounds
        self.res = SEARCH_RESOLUTION
        self.map = np.zeros((self.res))
    
    def get_coord(self, current_coord):
        """
            Obtener siguiente coordenada

            Seleccione una coordenada con un objeto encontrado o el mas cercano sin marcar

            Parametros
            ----------
            current_coord : float
                Coordenada actual

            Returns
            -------
            float
                Coordenada siguiente
        """

        indices = np.where(self.map == 1)[0]
        if len(indices) > 0:
            return self._get_coord(indices[0])
        
        indices = np.where(self.map == 0)[0]
        if len(indices) == 0:
            return current_coord
        current_index = self._get_index(current_coord)
        distance = np.abs(indices - current_index)
        nearest_index = indices[np.argmin(distance)]
        return self._get_coord(nearest_index)
            
    def mark_coord(self, coord, is_found):
        """
            Marcar resultado de verificacion por coordenada

            Parametros
            ----------
            coord : float
                Coordenada
            is_found : bool
                Se encuentro el objeto
        """

        index = self._get_index(coord)
        self.map[index] = 1 if is_found else -1

    def reset(self):
        """
            Restablecer resultados de la prueba
        """

        self.map = np.zeros((self.res))

    def _get_index(self, coord):
        index = self.res * (coord - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
        return np.clip(int(index), 0, self.res-1)

    def _get_coord(self, index):
        return (self.bounds[1] - self.bounds[0]) * index / self.res + self.bounds[0]

def find_and_approach(robot, detector, control_network):
    """
    Busca un objetivo a lo largo del area de trabajo y se acerca a el.
    No funciona en una escena estatica.

    Parametros
    ----------
    robot : Robot
        Robot controlado conectado al escenario
    detector : ObjectDetector
        Detector de objetos
    control_network : tf.keras.Model
        Red neuronal de control entrenada: estado -> accion

    Retorno
    -------
    bool
        Se alcanzo el objetivo
    """

    # Reduccion a valores estandar de todas las coordenadas del robot, excepto la primera
    next_pos = robot.default_pos.copy()
    next_pos[0] = robot.get_adometry_feedback()[0]
    robot.move(next_pos)

    searcher = Searcher(robot.joint_ranges[0,:])
    step = 0
    is_searching = True
    while True:
        state = network_utils.get_state(robot, detector)
        area = state[3]*state[4] # Area del rectangulo objetivo
        if area > 0.2:
            return True
        elif step >= 35:
            return False
        else:
            step+=1
            is_target_found = area > 0.001
            pos = network_utils.extract_pos(state)
            #print('posI:',pos)
            if is_searching:
                # Actualizar el mapa de presencia del objetivo solo cuando esta controlado por el buscador,
                # que garantiza los valores estandar de todas las coordenadas del robot, excepto la primera
                searcher.mark_coord(pos[0], is_target_found)
            
            # Calculo de la siguiente coordenada
            if is_target_found:
                action = control_network(np.expand_dims(state, axis=0)).numpy()
                action = np.squeeze(action, axis=0)
                next_pos = robot.clip_position(pos + action)
                #print('posEnc:',pos)
                is_searching = False
            else:
                next_pos = robot.default_pos.copy()
                next_pos[0] = searcher.get_coord(pos[0])
                is_searching = True

            is_pos_reached = robot.move(next_pos)
            if (not is_pos_reached) and is_searching:
                # El sector es inaccesible, hacemos la suposición de que no hay objetivo
                searcher.mark_coord(next_pos[0], False)
    return False