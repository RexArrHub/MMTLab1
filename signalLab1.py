import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal.windows import gaussian
from scipy.signal import ShortTimeFFT

# Загрузка аудиофайла
input_file_path = "model_signal.wav"
fd, data = read(input_file_path)

# Для стерео сигнала один канал (правый)
if len(data.shape) > 1:
    data = data[:, 1]

# Нормализация сигнала
max_amplitude = np.max(np.abs(data))
if max_amplitude > 0:
    data = data / max_amplitude

# Модуль построения амплитудного спектра
N = len(data)  # Количество отсчетов
Spectr_input = np.fft.fft(data)  # Быстрое преобразование Фурье
AS_input = np.abs(Spectr_input)  # Амплитудный спектр
eps = np.max(AS_input) * 1.0e-9  # Малое значение для избежания log(0)
S_dB_input = 20 * np.log10(AS_input + eps)  # Спектр в дБ

# Набор частот
f = np.arange(0, fd / 2, fd / N)
S_dB_input = S_dB_input[:len(f)]  # Выравниваем длины

# Построение графика амплитудного спектра
plt.figure(figsize=(10, 6))
plt.semilogx(f, S_dB_input)  # График в полулогарифмическом масштабе
plt.grid(True)
plt.minorticks_on()  # Отобразить мелкую сетку на лог.масштабе
plt.grid(True, which='major', color='#444', linewidth=1)
plt.grid(True, which='minor', color='#aaa', ls=':')

# Зададим лимиты для осей, по вертикали - кратно 20
Max_dB = np.ceil(np.max(S_dB_input) / 20) * 20
plt.axis([10, fd / 2, Max_dB - 100, Max_dB])  # Ограничим оси

plt.xlabel('Частота (Гц)')
plt.ylabel('Уровень (дБ)')
plt.title('Амплитудный спектр')
plt.show()

# Параметры для спектрограммы
T = N / fd  # Общая длительность сигнала в секундах

# Стандартное отклонение для Гауссова окна (в отсчетах)
g_std = 0.2 * fd

# Симметричное Гауссово окно
wind = gaussian(round(2 * g_std), std=g_std, sym=True)

# Создание объекта ShortTimeFFT
SFT = ShortTimeFFT(wind, hop=round(0.1 * fd), fs=fd, scale_to='magnitude')

# Вычисление STFT
Sx = SFT.stft(data)  # Выполняем STFT

# Построение графика спектрограммы
fig1, ax1 = plt.subplots(figsize=(10, 6))
t_lo, t_hi = SFT.extent(N)[:2]  # Временной диапазон графика
ax1.set_title(rf"Спектрограмма STFT (Окно Гаусса длительностью {SFT.m_num * SFT.T:g}$\,s$," +
              rf"$\sigma_t={g_std * SFT.T}\,$s)")
ax1.set(xlabel=f"Время $t$ в секундах ({SFT.p_num(N)} срезов," +
               rf"$\Delta t = {SFT.delta_t:g}\,$s)",
        ylabel=f"Частота $f$ в Гц ({SFT.f_pts} бинов, " +
               rf"$\Delta f = {SFT.delta_f:g}\,$Гц)",
        xlim=(t_lo, t_hi))
epss = np.max(abs(Sx)) * 1e-6
im1 = ax1.imshow(20 * np.log10(abs(Sx) + epss),
                 origin='lower', aspect='auto',
                 extent=SFT.extent(N), cmap='viridis')
fig1.colorbar(im1, label="Амплитуда $|S_x(t, f)|, dB $")

ax1.semilogy()
ax1.set_xlim([0, T])  # Ограничение по времени
ax1.set_ylim([10, fd / 2])  # Ограничение по частоте

# Отображение основной и дополнительной сетки
ax1.grid(which='major', color='#bbbbbb', linewidth=0.5)
ax1.grid(which='minor', color='#999999', linestyle=':', linewidth=0.5)
ax1.minorticks_on()
plt.show()