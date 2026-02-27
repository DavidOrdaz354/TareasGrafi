import cv2
import numpy as np

def procesar_frutas():
    path = r"C:\Users\death\Downloads\frutas.png"
    
    img = cv2.imread(path)
    
    #2. Covertir al esapcio de color HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    if img is None:
        print(f"ERROR: No se encontró la imagen en: {path}")
        print("Verifica que el nombre del archivo sea exactamente 'frutas.png'")
        return

    img = cv2.resize(img, (800, 600))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    print("--- MENU DE SELECCION ---")
    print("Opciones: rojo, verde, amarillo")
    seleccion = input("¿Qué color deseas analizar?: ").lower().strip()

    if seleccion == 'rojo':
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 100, 20])
        upper2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = mask1 + mask2
        
    elif seleccion == 'verde':
        lower = np.array([35, 50, 20])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
    elif seleccion == 'amarillo':
        lower = np.array([20, 100, 100])
        upper = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
    else:
        print("Color no reconocido. Intenta de nuevo.")
        return

    kernel = np.ones((5,5), np.uint8)
    mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_limpia = cv2.morphologyEx(mask_limpia, cv2.MORPH_CLOSE, kernel)

    resultado_color = cv2.bitwise_and(img, img, mask=mask_limpia)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_limpia, connectivity=8)

    print(f"\n--- REPORTE PARA: {seleccion.upper()} ---")
    frutas_reales = 0
    area_minima = 500

    print(f"{'ID Región':<10} | {'Área (px)':<10} | {'Estado'}")
    print("-" * 35)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        estado = "Ruido"
        if area > area_minima:
            frutas_reales += 1
            estado = "Fruta Detectada"
        print(f"{i:<10} | {area:<10} | {estado}")

    print("-" * 35)
    print(f"TOTAL FRUTAS DETECTADAS: {frutas_reales}")

    cv2.imshow('0. Imagen Original', img)
    cv2.imshow("1. Imagen HSV", hsv)
    cv2.imshow('2. Mascara Cruda (Con Ruido)', mask)
    cv2.imshow('3. Mascara Final (Procesada)', mask_limpia)
    cv2.imshow('4. Resultado a Color', resultado_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    procesar_frutas()