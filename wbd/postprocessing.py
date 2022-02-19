"""
Image postprocessing module.
"""
import os

from typing import Union, Tuple
from time import time

import numpy as np
import cv2
from ds_utils import load, pack_rgb, unpack_rgb
from scipy.cluster.vq import kmeans, vq
from PIL import Image
from wbdlogger import WBDLogger

logger = WBDLogger()


def postprocessing(output_path: str, crop_weights: list, tmp_dir: str):
    """
    Постобработка включает 2 этапа:
       * Обрезка высветленного изображения
       * Перевод записей на белый фон с удалением шумов

    :param output_path: путь для итогового изображения.
        Будет модифицирован добавлением префикса postprocessing к имени файла.
    :param crop_weights: веса для обрезки кадра.
    :param tmp_dir: временная директория для хранения промежуточных этапов постобработки
    """
    logger.info("Начинаю постобработку")
    # Prepare image - crop it, to leave only the whiteboard
    img = cv2.imread(output_path)

    # crop_weights[0] - up/down coefficients.
    # crop_weights[1] - left/right coefficients.
    original_shape: tuple = img.shape
    crop_img = img[crop_weights[0][0]:original_shape[0] + crop_weights[0][1],
               crop_weights[1][0]:original_shape[1] + crop_weights[1][1]]

    tmp_filename = os.path.join(tmp_dir, f'tmp_{str(int(time()))}.png')
    cv2.imwrite(tmp_filename, crop_img)
    logger.info("Записан временный файл с обрезанным изображением")

    # Postprocessing
    img_np = load(tmp_filename)
    if img_np is None:
        raise IOError

    sample: np.ndarray = sample_pixels(img_np)
    palette: np.ndarray = get_palette(sample)
    _back_indx: int = output_path.rfind('/') + 1

    labels = apply_palette(img_np, palette)
    output_filename = output_path[:_back_indx] + 'postprocessing_' + output_path[_back_indx:]
    save(output_filename, labels, palette, (300, 300))

    logger.info(f"Postprocessing окончен. Изображение сохранено по адресу: {output_filename}")
    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    logger.info("Увеличиваем чёткость изображения")
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def apply_brightness_contrast(input_img: np.ndarray, brightness: Union[int, float] = 0,
                              contrast: Union[int, float] = 0):
    """
    Функция увеличивает контраст и яркость исходного изображения без ухудшения качества пикселей.

    :param input_img: исходное изображение в формате массива.
    :param brightness: коэффициент яркости.
    :param contrast: коэффициент контраста.
    """
    if brightness != 0:
        logger.info(f"Увеличиваем яркость на {brightness}%")

        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        logger.info(f"Увеличиваем контраст на {contrast}%")

        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def sharpen(input_filename: str) -> str:
    im = cv2.imread(input_filename)
    im_bw = cv2.bilateralFilter(im, 9, 75, 75)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    im_bw = cv2.filter2D(im_bw, -1, kernel)

    out_filename = f'tmp/tmp_{int(time())}.png'
    cv2.imwrite(out_filename, im_bw)

    return out_filename


def save(output_filename: str, labels, palette: np.ndarray, dpi: tuple):
    """
    Сохраняет label/palette пары как индексированное PNG изображение.
    Дополнительно насыщает палитру, сопоставляя наименьший цветовой
    компонент нулю, а наибольший - 255, а также дополнительно устанавливает
    цвет фона на чистый белый.

    :param output_filename:
    :param labels:
    :param palette: цветовая палитра
    :param dpi: dpi выходного изображения
    :return:
    """
    # Saturate img
    logger.info("Насыщение палитры - подбор более ярких цветов")
    palette = palette.astype(np.float32)
    pmin = palette.min()
    pmax = palette.max()
    palette = 255 * (palette - pmin) / (pmax - pmin)
    palette = palette.astype(np.uint8)

    # Make white background
    logger.info("Перевод фона в белый цвет")
    palette = palette.copy()
    palette[0] = (250, 250, 250)

    output_img = Image.fromarray(labels, 'P')
    output_img.putpalette(palette.flatten())
    if output_img.mode != 'RGB':
        output_img = output_img.convert('RGB')
    output_img.save(output_filename, dpi=dpi)


def apply_palette(img, palette):
    """
    Apply the pallete to the given image.
    The first step is to set all background pixels to the background color;
    then, nearest-neighbor matching is used to map each foreground color to the closest one in the palette.
    """
    logger.info("Применение палитры")
    bg_color = tuple(palette[0])

    fg_mask = get_foreground_mask(bg_color, img.reshape(-1, 3))
    orig_shape = img.shape
    logger.info("Фоновые пиксели отмечены")

    pixels = img.reshape((-1, 3))
    fg_mask = fg_mask.flatten()
    num_pixels = pixels.shape[0]

    logger.info("Сопоставление каждого цвета переднего плана с ближайшим цветом в палитре")
    labels = np.zeros(num_pixels, dtype=np.uint8)
    labels[fg_mask], _ = vq(pixels[fg_mask], palette)

    return labels.reshape(orig_shape[:-1])


def get_palette(sample: np.ndarray, num_colors: int = 20, kmeans_iter: int = 200) -> np.ndarray:
    """
    Позволяет получить палитру из num_colors наиболее часто встречающихся цветов

    :param sample: случайная выборка из исходного изображения
    :param num_colors: количество желаемых цветов в палитре
    :param kmeans_iter: количество итераций в алгоритме k-means

    :return: палитра цветов, найденных в sample, в количестве num_colors
    """
    logger.info("Подготовка палитры")

    bg_color: tuple = get_bg_color(img=sample, bits_per_channel=6)
    logger.info(f"Фон определён: {str(bg_color)}")

    fg_mask: np.ndarray = get_foreground_mask(bg_color, sample)

    centers, _ = kmeans(sample[fg_mask].astype(np.float32),
                        num_colors - 1,
                        iter=kmeans_iter)

    palette: np.ndarray = np.vstack((bg_color, centers)).astype(np.uint8)

    return palette


def get_foreground_mask(bg_color: tuple, sample: np.ndarray) -> np.ndarray:
    """
    Мы можем пометить пиксель как принадлежащий к фону, если выполняется один из этих критериев:
        * hsv value отличается от bg_color более чем на 0.25-0.3;
        * hsv saturation отличается от bg_color более чем на 0.2.

    :param bg_color: цвет фона в RGB
    :param sample: случайная выборка пикселей из исходного изображения

    :return: принадлежность пикселя из sample к фоновому
    """
    import colorsys

    _, s_bg, v_bg = colorsys.rgb_to_hsv(bg_color[0] / 255, bg_color[1] / 255, bg_color[2] / 255)

    s_smpl = list()
    v_smpl = list()

    for el in sample:
        _, s_s, v_s = colorsys.rgb_to_hsv(el[0] / 255, el[1] / 255, el[2] / 255)
        s_smpl.append(s_s)
        v_smpl.append(v_s)

    s_diff = np.abs(s_bg - np.array(s_smpl))
    v_diff = np.abs(v_bg - np.array(v_smpl))

    return ((v_diff >= 0.25) |
            (s_diff >= 0.2))


def sample_pixels(img: np.ndarray, pixels_percent: int = 40) -> np.ndarray:
    """
    Выбирает в случайном порядке набор пикселей, в размере options % от переданного img

    :param img: изображение в матричном формате
    :param pixels_percent: количество пикселей в %

    :return: выборка пикселей в размере pixels_percent % от исходного img
    """
    logger.info("Готовим sample для определения фона")
    pixels = img.reshape((-1, 3))
    num_pixels = pixels.shape[0]

    num_samples = int(num_pixels * pixels_percent / 100)
    idx = np.arange(num_pixels)

    np.random.shuffle(idx)
    return pixels[idx[:num_samples]]


def get_bg_color(img: np.ndarray, bits_per_channel=6) -> Tuple[int]:
    """
    Получает цвет фона из изображения или массива цветов RGB,
    группируя похожие цвета в ячейки и находя наиболее часто встречающийся.

    :param img: изображение в матричном формате
    :param bits_per_channel: желаемое количество бит на канал. По умолчанию - 6

    :return: 24-х битный RGB триплет с фоновым цветом
    """
    logger.info("Поиск цвета фона")
    # Проверка трёхслойности изображения
    assert img.shape[-1] == 3

    quantized = quantize(img, bits_per_channel).astype(int)
    packed = pack_rgb(quantized)

    unique, counts = np.unique(packed, return_counts=True)

    packed_mode = unique[counts.argmax()]

    return tuple(unpack_rgb(packed_mode))


def quantize(img: np.ndarray, bits_per_channel: int = 6):
    """
    Уменьшает количество бит на канал в переданном img.
    По сути, уменьшая битовую глубину, мы группируем похожие пиксели в более крупные «ячейки»,
    что упрощает поиск цвета фона.
    """
    assert img.dtype == np.uint8

    shift = 8 - bits_per_channel
    halfbin = (1 << shift) >> 1

    return ((img.astype(int) >> shift) << shift) + halfbin
