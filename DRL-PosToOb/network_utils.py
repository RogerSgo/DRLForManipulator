import tensorflow as tf
import numpy as np

STATE_SIZE = 11 # dimension del espacio de estado
ACTION_SIZE = 6 # dimension del espacio de acción

def create_qnetwork():
    """
        Crear una red neuronal-critica.

        Retorno
        -------
        tf.keras.Model
            Modelo que toma [accion, estado] y devuelve una puntuacion numerica
    """
    input1 = tf.keras.Input(shape=(ACTION_SIZE))
    input2 = tf.keras.Input(shape=(STATE_SIZE))
    inputs = [input1, input2]
    x = tf.concat(inputs, axis=1)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(64, activation="sigmoid")(x)
    output = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model

def create_control_network():
    """
        Crear una red neuronal-actor.

        Retorno
        -------
        tf.keras.Model
            Un modelo que toma un estado y devuelve una accion - cambio por coordenadas
    """

    input = tf.keras.Input(shape=(STATE_SIZE))
    x = tf.concat(input, axis=1)
    x = tf.keras.layers.Dense(64)(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)
    
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("sigmoid")(x)

    output = tf.keras.layers.Dense(ACTION_SIZE, activation="tanh")(x) / 2.0
    model = tf.keras.Model(inputs=input, outputs=output)
    return model

def update_model(old, new, rate):
    """
        Actualizar pesos en red neuronal

        Parametros
        ----------
        old : tf.keras.Model
            Modelo cuyos pesos necesitan ser actualizados
        new : tf.keras.Model
            El modelo cuyos pesos usar para la actualizacion
        rate : float
            Factor de suavizado exponencial - tasa de actualizacion
    """
    new_w = new.get_weights()
    w = old.get_weights()
    for i in range(len(new_w)):
        w[i] = w[i]*(1-rate) + new_w[i] * rate
    old.set_weights(w)

def prepare_state_features(pos, depth_map, obj_rect):
    """
        Preparar vector de estado

        Parametros
        ----------
        pos : ndarray
            Coordenadas generalizadas actuales
        depth_map : ndarray
            Mapa de profundidad estereo
        obj_rect : list
            Rectangulo [x,y,ancho,alto] objeto de marco en la imagen

        Retorno
        -------
        ndarray
            vector de estado
    """
    X = np.zeros((STATE_SIZE), dtype=np.float32)
    x,y,w,h = obj_rect

    if w != 0:
        width = depth_map.shape[0]
        height = depth_map.shape[1]

        X[0] = np.mean(depth_map[x:x+w, y:y+h])
        if X[0] < 0:
            X[0] = 0
        X[1] = (x + w - width) / width / 2
        X[2] = (y + h - height) / height / 2
        X[3] = w / width
        X[4] = h / height
    X[5:] = pos
    return X

def get_state(robot, detector):
    """
        Obtener estado.
        Un envoltorio para determinar el estado actual utilizando la retroalimentacion del robot y el detector.

        Parametros
        ----------
        robot : Robot
            Robot controlado
        detector : ObjectDetector
            Detector de objetos utilizado

        Retorno
        -------
        ndarray
            Vector de estado
    """
    img, map = robot.get_vision_feedback()
    pos = robot.get_adometry_feedback()
    obj_loc = detector.detect(img)
    return prepare_state_features(pos, map, obj_loc)

def extract_pos(state):
    """
        Extraer coordenadas generalizadas y vectores de estado

        Parametros
        ----------
        state : ndarray
            Vector de estado

        Retorno
        -------
        ndarray
            Coordenadas generalizadas
    """
    if len(state.shape) == 2:
        return state[:, 5:]
    else:
        return state[5:]