# CodeAlign-Runtime

sudo nvidia-ctk runtime configure --runtime=docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker

docker build -t codealign-runtime .
docker run --gpus all -it --rm --env-file .env -v $(pwd):/app codealign-runtime
python scripts/tu_script_baseline.py

// thrust::host_vector y thrust::device_vector??

ncu --set default ./gemv_naive 
==WARNING== No metrics to collect found in sections.
==PROF== Connected to process 32564 (/home/sergas/CodeAlign-Runtime/src/gemv_naive)
==PROF== Profiling "gemv_naive_kernel" - 0: 0%....50%....100% - 1 pass
1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 
==PROF== Disconnected from process 32564
[32564] gemv_naive@127.0.0.1
  gemv_naive_kernel(const float *, const float *, float *, int, int) (4, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 12.0

con sudo ->
==PROF== Connected to process 45256 (/home/sergas/CodeAlign-Runtime/src/gemv_naive)
==ERROR== Profiling is not supported on device 0. To find out supported GPUs refer --list-chips option.
1024 1024 1024 1024 1024 1024 1024 1024 1024 1024 
Execution time: 0.703872 ms
Total Bandwith 0.0597054 GB/s
==PROF== Disconnected from process 45256
==WARNING== No kernels were profiled.
==WARNING== Profiling kernels launched by child processes requires the --target-processes all option.

3. Formas de matriz reales de un LLMLa matriz actual es de 1024x1024. Para una GPU moderna, esto es tan pequeño que el kernel termina antes de que la tarjeta gráfica se entere, lo que significa que gran parte de tu tiempo medido es solo latencia de comunicación entre la CPU y la GPU.El documento indica que debes implementar y probar las distintas formas de GEMV que aparecen de verdad en el modelo. Como mencionaste usar modelos de 0.5B a 1.5B, probemos con las dimensiones reales de un bloque típico (como Qwen 0.5B):  Proyección QKV: rows = 896, cols = 896Proyección MLP (Up): rows = 4864, cols = 896

El Problema: Accesos a Memoria no Coalescentes (Uncoalesced)Para entender la coalescencia, tienes que saber cómo lee la memoria una GPU físicamente.La GPU no lee los datos flotante a flotante (de 4 en 4 bytes). Cuando un hilo pide un dato a la VRAM, el controlador de memoria va y trae un "vagón de tren" completo de datos consecutivos (normalmente de 32 o 128 bytes).Los hilos en CUDA se ejecutan en grupos de 32 llamados Warps.Acceso Coalescente (Lo ideal): Si el hilo 0 pide el índice 0, el hilo 1 el índice 1, el hilo 2 el índice 2... todos caen dentro del mismo "vagón de tren". La GPU hace un solo viaje a la memoria y alimenta a los 32 hilos de golpe. Eficiencia máxima.Acceso No Coalescente (Nuestro kernel actual): En tu kernel naive, al hilo 0 le asignaste la fila 0 y al hilo 1 la fila 1. El primer elemento de la fila 0 está en el índice 0, pero el primer elemento de la fila 1 está en el índice 896. Están separadísimos en la memoria física. La GPU se ve obligada a traer un vagón entero para darle un solo flotante al hilo 0, tirar el resto del vagón, y traer otro vagón completamente distinto para el hilo 1.Estás saturando el controlador de memoria con viajes inútiles. Por eso tu ancho de banda aprovechado ronda los 190 GB/s en lugar de acercarse al límite máximo de tu tarjeta.Introducción al Nivel 2: Domando la Jerarquía de MemoriaEl objetivo principal de esta nueva fase es rediseñar nuestro algoritmo para que la lectura de memoria sea colaborativa, demostrando así que entiendes la jerarquía de memoria de la GPU. Este es exactamente el tipo de conocimiento que busca JetBrains y que se pregunta en entrevistas técnicas.  Para lograr que el rendimiento se dispare y se acerque al de cuBLAS, aplicaremos tres técnicas fundamentales de optimización:  Cambiar la asignación de hilos (Coalescencia): En lugar de que un solo hilo calcule una fila entera, haremos que un warp entero (32 hilos) o un bloque colabore para calcular partes de la misma fila simultáneamente. Así, leerán posiciones de memoria contiguas en el mismo ciclo de reloj.Cargas Vectorizadas (float4): Modificaremos el código para que cada hilo no lea un float de 4 bytes, sino una estructura float4 (16 bytes) por instrucción. Es como usar una pala más grande para sacar los datos de la memoria.  Warp Shuffles para la reducción final: Como ahora tendremos a varios hilos calculando sumas parciales de la misma fila, necesitamos sumarlas todas al final. En lugar de usar la memoria compartida convencional para esto, utilizaremos las primitivas intrínsecas de CUDA (__shfl_down_sync), que permiten a los hilos pasarse datos directamente a través de los registros del procesador sin tocar la memoria, siendo la técnica estándar en kernels modernos.  

-----NAIVE KERNEL-----
Average Execution Time: 0.092016 ms
Average Bandwidth 189.702 GB/s
-----OPTIMIZED KERNEL-----
Average Execution Time: 0.0128423 ms
Average Bandwidth 1359.23 GB/s

Ese error Profiling is not supported ocurre porque la versión de NVIDIA Nsight Compute (ncu) que viene con tu toolkit de CUDA aún no tiene los perfiles de hardware actualizados para la ultimísima generación de chips de la serie 50. Es un problema clásico de estar en la vanguardia del hardware.Pero no te preocupes, no necesitamos que ncu funcione para cumplir con este paso del roadmap. De hecho, tu situación nos acaba de regalar el mejor insight técnico que puedes poner en el README de tu portfolio.El Insight de Oro para tu Entrevista (Caché vs VRAM)El objetivo del roadmap era reportar el porcentaje del ancho de banda pico alcanzado para demostrar que entiendes la jerarquía de memoria de la GPU. Como ncu no funciona, lo calcularemos analíticamente con los datos de tu ejecución limpia (la que te dio 1359 GB/s).  Aquí es donde entra la física del hardware:El tamaño de tu problema: Tu matriz es de 4864 x 896 flotantes. A 4 bytes por flotante, son exactamente 17.4 MB de datos.Tu hardware: Una RTX 5070 Ti tiene una memoria caché L2 masiva (probablemente rondando los 48 MB - 64 MB). Su ancho de banda de memoria principal (VRAM GDDR7) teóricamente rondará los 700-800 GB/s.El fenómeno: Tu bucle en main.cpp ejecuta el kernel 100 veces sobre los mismos datos. En la primera iteración, la GPU trae los 17.4 MB de la VRAM (lenta). Pero para las 99 iteraciones siguientes, ¡la matriz entera se ha quedado guardada en la caché L2 ultrarrápida del chip!Ese 1359 GB/s que mediste no es el ancho de banda de la memoria VRAM, ¡es el ancho de banda de la caché L2 de tu RTX 5070 Ti!Si en una entrevista técnica para JetBrains explicas que tu benchmark naive dio ~189 GB/s (atascado por mala coalescencia), pero el optimizado se disparó a 1.3 TB/s porque lograste saturar el bus y, además, los 17 MB del modelo cabían en la caché L2, te contratarán en el acto. Esto demuestra una comprensión profunda de sistemas ML, mucho más allá de simplemente "escribir CUDA"