# Segmentación de Frutas usando Máscara HSV

**Alumno:** David Emanuel Ordaz Amezcua  
**Materia:** Graficación  
**Periodo:** Enero – Junio 2026  

---

## Objetivo

Aplicar el modelo de color HSV para segmentar objetos dentro de una imagen digital, analizando directamente la información contenida en una máscara binaria. Asimismo, identificar y contar regiones conectadas correspondientes a frutas, evaluar el impacto que tiene el ajuste del rango HSV sobre la calidad de la segmentación y reflexionar sobre las ventajas y limitaciones de este enfoque.

---

## Material

- Imagen proporcionada: `frutas.png`  
- Lenguaje de programación: Python  
- Librerías:
  - OpenCV  
  - NumPy  

---

## Contexto

El análisis se realizó únicamente sobre la máscara binaria generada a partir de un rango HSV.  
No se utilizaron contornos, rectángulos ni marcas sobre la imagen original.  
Todo el conteo y validación se llevó a cabo mediante el análisis matemático de regiones conectadas presentes en la máscara procesada.

---

# Actividad 1: Exploración del Espacio HSV (Color Verde)

Para esta práctica se seleccionó el color **VERDE**.  
Se ajustaron los valores de Matiz (H), Saturación (S) y Valor (V) hasta lograr que la máscara detectara únicamente las frutas verdes presentes en la imagen, procurando reducir la detección de fondo u otros objetos.

---

## Capturas solicitadas

### Imagen original
![Imagen Original](imagenes/original.png)

![Mascara Cruda](imagenes/mascara_cruda.png)

![Mascara Final](imagenes/mascara_final.png)

![Resultado Color](imagenes/resultado_color.png)


---

## Reflexión

Cuando el rango HSV es demasiado estrecho, algunas partes de las frutas no son detectadas, provocando huecos negros dentro de las regiones blancas. Esto ocurre principalmente en zonas donde existen sombras o variaciones de iluminación.

Cuando el rango HSV es demasiado amplio, la máscara comienza a detectar objetos que no pertenecen a las frutas, como hojas, tallos o incluso pequeñas zonas del fondo, lo que incrementa considerablemente el ruido.

Por lo tanto, es necesario encontrar un equilibrio en el rango HSV para obtener una segmentación adecuada.

---

# Actividad 2: Limpieza de Ruido

Se analizó primero la máscara cruda, donde se observaron múltiples puntos blancos aislados y pequeñas regiones no deseadas.

Posteriormente se aplicaron operaciones morfológicas de **apertura** y **cierre** para eliminar ruido pequeño y rellenar huecos dentro de las frutas.

### ¿Qué tipo de ruido aparece?

- Píxeles aislados (ruido tipo sal y pimienta).  
- Regiones correspondientes a hojas y tallos verdes.

### ¿Por qué es necesario eliminarlo antes del conteo?

Porque cada región blanca es interpretada como un objeto independiente. Si el ruido no se elimina, el sistema contaría muchos objetos inexistentes, produciendo un resultado incorrecto.

---

# Actividad 3: Conteo de Regiones

Después de aplicar un filtro de área mínima de 500 píxeles:

**TOTAL DE FRUTAS VERDES DETECTADAS: 21**

Las regiones con área menor al umbral fueron clasificadas como ruido.

El conteo se realizó únicamente sobre la máscara binaria, sin utilizar la imagen original como referencia visual.

---

# Actividad 4: Comparación entre Colores

Aunque el análisis principal se realizó con el color verde, se probó el algoritmo con los otros colores para comparar el comportamiento del sistema.

| Color     | Número Detectado | Observaciones |
|----------|-----------------|---------------|
| Rojo     | 8  | Presenta huecos debido a brillos en la superficie de las frutas. |
| Verde    | 21 | Alto ruido por presencia de hojas y tallos del mismo color. |
| Amarillo | 10 | Segmentación más estable y con menos interferencias. |

### ¿Qué color fue más fácil segmentar?

El amarillo, debido a que tiene menos interferencias con otros objetos de la imagen.

### ¿Cuál presentó más ruido?

El verde, ya que existen otros elementos verdes en la imagen.

### ¿Por qué?

Porque el algoritmo se basa únicamente en el color y no distingue entre fruta y hoja si ambos comparten tonalidades similares.

---

# Actividad 5: Análisis Crítico

### 1. ¿Por qué HSV es más adecuado que RGB para esta tarea?

HSV separa la información de color de la iluminación, lo que facilita segmentar objetos por su tono sin verse tan afectado por sombras o brillos.

### 2. ¿Cómo afecta la iluminación al canal V?

El canal V representa el brillo. Cambios de iluminación pueden hacer que partes de las frutas no entren en el rango, generando pérdidas de información.

### 3. ¿Qué sucede si dos frutas tienen tonos similares?

El sistema puede fusionarlas en una sola región, provocando errores en el conteo.

### 4. ¿Qué limitaciones tiene la segmentación por color?

No considera forma ni textura, depende de la iluminación y puede confundir objetos con colores similares.

---

# Conclusión Final

La segmentación utilizando el modelo HSV demostró ser una técnica efectiva para detectar frutas dentro de una imagen digital, especialmente cuando se cuenta con colores bien definidos y un fondo relativamente uniforme. A lo largo de la práctica se observó que la correcta selección del rango HSV es un factor crítico, ya que un rango demasiado estrecho provoca pérdida de información y un rango demasiado amplio genera ruido.

El caso del color verde evidenció una de las principales limitaciones de la segmentación por color: la dificultad para diferenciar objetos distintos que comparten tonalidades similares, como frutas y hojas. Esto demuestra que, aunque HSV es una herramienta poderosa, no es suficiente por sí sola para resolver problemas de segmentación complejos.

Las operaciones morfológicas y el filtrado por área resultaron esenciales para mejorar la calidad de la máscara y obtener un conteo confiable. Finalmente, se concluye que la segmentación por color debe considerarse como una etapa inicial dentro de un sistema de visión artificial más completo, el cual podría complementarse con análisis de forma, textura o técnicas de aprendizaje automático para obtener mejores resultados.

---

**Fin del documento**
