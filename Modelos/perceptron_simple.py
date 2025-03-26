import numpy as np

def paso(x):
    return 1 if x >= 0 else 0

def perceptron(u, w0, w1, w2):
    # Tabla de AND
    datos = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1]
    ])

    x0 = 1
    epoch = 0

    # Inicialización
    error = True

    while error:
        error = False  # Reiniciar el indicador de error para la nueva época
        print(f"Época {epoch}")

        for fila in datos:
            x1, x2, resul_esperado = fila
            
            y = (w0 * x0) + (w1 * x1) + (w2 * x2)
            y = paso(y)
            
            if y != resul_esperado:
                delta = resul_esperado - y
                w0 += u * delta * x0
                w1 += u * delta * x1
                w2 += u * delta * x2
                w0 = round(w0, 2)
                w1 = round(w1, 2)
                w2 = round(w2, 2)
                error = True
                break  # Salir del bucle al encontrar el primer error

            print(f"x1={x1}, x2={x2}, esperado={resul_esperado}, salida={y}")
            print(f"Pesos: w0={w0}, w1={w1}, w2={w2}")

        print("-" * 30)

        if not error:
            # Si no hay error el resultado ha sido el esperado y salimos del bucle
            break
        
        #Actualizar época
        epoch += 1

    return w0, w1, w2

u = 0.1  # Tasa de aprendizaje
w0 = 0   # Peso inicial del bias
w1 = 1   # Peso inicial para x1
w2 = 1   # Peso inicial para x2

# Ejecutar el perceptrón
final_w0, final_w1, final_w2 = perceptron(u, w0, w1, w2)
print(f"Pesos finales: w0={final_w0}, w1={final_w1}, w2={final_w2}")