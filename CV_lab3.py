from time import time
import numpy as np
import cv2

# Открываем видеофайл
video_path = 'IMG_0823.MOV'
cap = cv2.VideoCapture(video_path)

# Проверяем, успешно ли открылся файл
if not cap.isOpened():
    print("Ошибка: Не удалось открыть видеофайл.")
    exit()

c = 0
background = None
threshold_background_difference = 20  # Порог для разницы между фоном и текущим кадром

# Читаем и обрабатываем кадры
start_time = time()
c = 0
while cap.isOpened():

    ret, frame = cap.read()  # чтение кадра из видео

    if c < 100:
        c += 1
        continue  # Пропускаем первые 100 кадров (для стабилизации)

    if not ret:
        print('Видео не загрузилось, либо закончилось')
        break

    # Обрезаем и изменяем размер кадра
    frame = frame[300:1920 - 700, 0:1080]  # Обрезка кадра
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))  # Уменьшение размера
    cv2.imshow('Video', frame)  # Показываем кадр с исходным видеопотоком

    frame_with_contours = frame.copy()  # Копия кадра для отрисовки центра объекта

    # Преобразуем кадр в оттенки серого для ускорения вычислений и избежания зависимости от цветовых каналов
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Инициализация фона (первый кадр)
    if background is None:
        background = frame
    # Усреднение фона в течение первых 5 секунд
    elif time() - start_time < 5:
        background = np.add(background, cv2.convertScaleAbs(np.subtract(background, frame), alpha=0.01, beta=0))

    cv2.imshow('Background', background)  # Показываем текущий фон

    # Вычисляем разницу между фоном и текущим кадром
    difference = np.abs(np.subtract(background, frame))
    print(difference)
    # Создаем маску на основе порога
    mask = np.where(difference > threshold_background_difference, 255, 0).astype(np.uint8)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)  # Бинаризация маски

    # Морфологические операции для фильтрации шума, теней на фоне итд, увеличиваем действительно весомые участки
    mask = cv2.erode(mask, kernel=(10, 10), iterations=5)  # Эрозия
    mask = cv2.erode(mask, kernel=(5, 5), iterations=3)  # Эрозия
    mask = cv2.erode(mask, kernel=(3, 3), iterations=3)  # Эрозия
    mask = cv2.dilate(mask, kernel=(5, 5), iterations=3)  # Дилатация

    cv2.imshow('Mask', mask)  # Показываем маску

    # Находим контуры на маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Выбираем контур с максимальной площадью
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)  # Вычисляем моменты контура

        # Находим центр контура
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Рисуем центр бОльшего контура на кадре
        cv2.circle(frame_with_contours, (cx, cy), 20, (0, 0, 255), 3)
        cv2.circle(frame_with_contours, (cx, cy), 50, (0, 0, 255), 3)  # и примерные границы объекта...

    cv2.imshow('Captured object', frame_with_contours)  # Показываем кадр с контуром

    # Выход по нажатию 'q'
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()