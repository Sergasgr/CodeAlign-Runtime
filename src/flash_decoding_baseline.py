# Lo muevo a scripts? Que valores los puedo meter a un config como el baseline_config.py en un flash_decoding_baseline_config.py
import torch
import torch.nn.functional as F
import math

import codealign_runtime_kernels # falta recompilar?

class HeadAttention:
    def __init__(self, D = 128, S = 1024):
        self.D = D
        self.S = S
        self.Q = torch.rand(1, 1, self.D, device='cuda')
        self.K = torch.rand(1, self.S, self.D, device='cuda')
        self.V = torch.rand(1, self.S, self.D, device='cuda')

    def attention_pytorch(self):
        scores = (self.Q @ self.K.transpose(-2, -1)) / math.sqrt(self.D)
        attn_weights = F.softmax(scores, dim=-1)
        O = attn_weights @ self.V
        return O

    def attention_custom(self):
        return codealign_runtime_kernels.flash_decoding_forward(self.Q, self.K, self.V)

def validate_correctness():
    head_attention = HeadAttention()
    O_pt = head_attention.attention_pytorch()
    O_custom = head_attention.attention_custom()
    diff = torch.max(torch.abs(O_pt - O_custom))
    return diff < 1e-2

def benchmark():
    print("\n--- BENCHMARK (Batch=1, KV-Cache Growth) ---")
    D = 128
    sequences = [256, 1024, 4096, 8192, 16384]
    attention_start = torch.cuda.Event(enable_timing=True)
    attention_end = torch.cuda.Event(enable_timing=True)
    
    for S in sequences:
        head_attention = HeadAttention(D, S)
        for _ in range(10): # Warmup
            head_attention.attention_pytorch()
            head_attention.attention_custom()
        for _ in range(100):  
            attention_start.record()
            head_attention.attention_pytorch()
            attention_end.record()
            torch.cuda.synchronize()
        for _ in range(100):  
            attention_start.record()
            head_attention.attention_custom()
            attention_end.record()
            torch.cuda.synchronize()
    pass  # Imprime el tiempo promedio por llamada de cada uno. Validate correctness???

if __name__ == "__main__":
    validate_correctness()
    benchmark()

"""
Validación Numérica (El arnés de C++)

Como establece la arquitectura del proyecto, la velocidad no importa si el código de salida es incorrecto. La matemática de punto flotante en GPU no es asociativa, por lo que sumar por bloques generará ligeras variaciones frente a sumar todo de golpe en PyTorch.

    Tolerancia estricta: Implementa en tu main.cpp (o en un script de Python usando Pybind11) una validación que calcule el error absoluto máximo entre tu salida d_O_final y la referencia matemática.

    Umbral: Debes asegurar explícitamente que el error absoluto máximo sea menor a 10−2 en fp16/fp32. Si el error es mayor, hay un fallo en la lógica de corrección exponencial del Online Softmax.

    Pruebas robustas: No valides con un solo caso. Llama a tu función de validación con múltiples semillas aleatorias y distintas longitudes de secuencia S (ej. 256, 1024, 2048) para asegurar que la división por chunks funciona incluso cuando la secuencia no es un múltiplo exacto de chunk_size.

Benchmarking contra el Baseline (Cierre del Nivel 4)

Una vez que el código compile y pase la validación numérica de 10−2, debes demostrar empíricamente por qué elegiste Flash-Decoding en lugar de FlashAttention para este caso específico de autocompletado de código en IDEs.

    Usa el mismo script baseline.py que diseñaste para el Nivel 0, midiendo exclusivamente la fase de decodificación (batch=1, un solo token nuevo contra un KV-cache en crecimiento).

    Registra el tiempo de ejecución (con eventos CUDA, descartando warmup) usando la atención estándar de PyTorch.

    Intercambia la función de PyTorch por tu flash_decoding_forward inyectado vía Pybind11 y repite la medición.

    Lo que deberías observar —y documentar en tu tabla final del README— es que mientras el rendimiento de PyTorch se degrada linealmente a medida que el KV-cache crece, tu TPOT (Time-Per-Output-Token) se mantiene casi plano, probando que paralelizar sobre la dimensión del caché fue el diseño arquitectónico correcto.
"""