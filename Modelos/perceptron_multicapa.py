import numpy as np

# Función de activación sigmoide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la sigmoide
def derivada_sigmoide(x):
    return x * (1 - x)

# Perceptrón multicapa
def perceptron_xor(entrenamiento, etiquetas, tasa_aprendizaje, epochs):
    # Inicializar pesos y bias
    np.random.seed(42)  # Para reproducibilidad
    n_entradas = entrenamiento.shape[1]
    n_oculta = 2  # Neuronas en la capa oculta
    n_salida = 1  # Neurona en la capa de salida

    # Pesos y bias entre capa de entrada y capa oculta
    pesos_entrada_oculta = np.random.uniform(-1, 1, (n_entradas, n_oculta))
    bias_oculta = np.random.uniform(-1, 1, (1, n_oculta))

    # Pesos y bias entre capa oculta y capa de salida
    pesos_oculta_salida = np.random.uniform(-1, 1, (n_oculta, n_salida))
    bias_salida = np.random.uniform(-1, 1, (1, n_salida))

    # Entrenamiento
    for epoch in range(epochs):
        # Forward pass
        # Capa oculta
        entrada_oculta = np.dot(entrenamiento, pesos_entrada_oculta) + bias_oculta
        salida_oculta = sigmoide(entrada_oculta)

        # Capa de salida
        entrada_salida = np.dot(salida_oculta, pesos_oculta_salida) + bias_salida
        salida = sigmoide(entrada_salida)

        # Cálculo del error
        error = etiquetas - salida

        if epoch % 1000 == 0:  # Mostrar error cada 1000 épocas
            print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

        # Backpropagation
        # Gradiente para la capa de salida
        gradiente_salida = error * derivada_sigmoide(salida)

        # Gradiente para la capa oculta
        error_oculta = gradiente_salida.dot(pesos_oculta_salida.T)
        gradiente_oculta = error_oculta * derivada_sigmoide(salida_oculta)

        # Actualización de pesos y bias
        pesos_oculta_salida += salida_oculta.T.dot(gradiente_salida) * tasa_aprendizaje
        bias_salida += np.sum(gradiente_salida, axis=0, keepdims=True) * tasa_aprendizaje

        pesos_entrada_oculta += entrenamiento.T.dot(gradiente_oculta) * tasa_aprendizaje
        bias_oculta += np.sum(gradiente_oculta, axis=0, keepdims=True) * tasa_aprendizaje

    return pesos_entrada_oculta, bias_oculta, pesos_oculta_salida, bias_salida

# Datos de entrada para XOR
entrenamiento = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Etiquetas de salida para XOR
etiquetas = np.array([[0], [1], [1], [0]])

# Parámetros del modelo
tasa_aprendizaje = 0.1
epochs = 10000

# Entrenar el modelo
pesos_entrada_oculta, bias_oculta, pesos_oculta_salida, bias_salida = perceptron_xor(entrenamiento, etiquetas, tasa_aprendizaje, epochs)

# Probar el modelo
def predecir(entrada):
    # Forward pass
    entrada_oculta = np.dot(entrada, pesos_entrada_oculta) + bias_oculta
    salida_oculta = sigmoide(entrada_oculta)
    entrada_salida = np.dot(salida_oculta, pesos_oculta_salida) + bias_salida
    salida = sigmoide(entrada_salida)
    return np.round(salida)

# Probar con los datos de entrenamiento
print("Resultados:")
for entrada, etiqueta_real in zip(entrenamiento, etiquetas):
    prediccion = predecir(np.array([entrada]))
    print(f"Entrada: {entrada}, Esperado: {etiqueta_real[0]}, Predicción: {prediccion[0][0]}")
