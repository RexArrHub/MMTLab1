import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

Fd = 44100  # частота дискретизации
T = 3  # длительность аудио в сек.
N = round(T * Fd)

def harm_wave(f, A, N):
    signal_s = A * np.sin(2 * np.pi * f * np.arange(N) / Fd)
    signal_c = A * np.cos(2 * np.pi * f * np.arange(N) / Fd)
    model_signal = np.vstack((signal_s, signal_c))
    return model_signal

def normalize_signal(stereo_signal):
    Norm = np.max(np.abs(stereo_signal))
    if Norm != 0:
        stereo_signal = stereo_signal / Norm
    return stereo_signal

f, A = 432, 1.0
model_harm = harm_wave(f, A, N)

# Огибающая амплитуды:
w = (1 - np.cos(2 * np.pi * np.arange(N) / (Fd * T))) ** 2

# Применение огибающей к каналам:
left_channel = model_harm[0] * w
right_channel = model_harm[1] * (1 - w)

# Объединение каналов в один массив для нормировки
stereo_signal = np.vstack((left_channel, right_channel))

# Нормировка сигнала
stereo_signal = normalize_signal(stereo_signal)

# Преобразование в int16 для сохранения в формате WAV
left_channel_int16 = (stereo_signal[0] * 32767).astype(np.int16)
right_channel_int16 = (stereo_signal[1] * 32767).astype(np.int16)

# Объединение каналов в один массив для записи
stereo_signal_int16 = np.column_stack((left_channel_int16, right_channel_int16))

# Сохранение в файл
output_file = "model_signal.wav"
sf.write(output_file, stereo_signal_int16, Fd)
print(f"Файл {output_file} сохранен")

# Временная ось для всего сигнала
time_axis = np.arange(len(stereo_signal[0])) / Fd

# Построение графика для каждого канала
plt.figure(figsize=(10, 6))
plt.plot(time_axis, stereo_signal[0], color='b', label='Левый канал', linewidth=1)
plt.plot(time_axis, stereo_signal[1], color='r', label='Правый канал', linewidth=1)
plt.title('Аудио сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid(linestyle='--', linewidth=0.5)
plt.legend()

# Увеличение масштаба для детального рассмотрения
zoom_start = 0.65  # Начало отрезка
zoom_end = 0.85  # Конец отрезка. Для лучшего масштаба поставить 0.1
plt.xlim(zoom_start, zoom_end)  # Устанавливаем границы по оси X
plt.show()