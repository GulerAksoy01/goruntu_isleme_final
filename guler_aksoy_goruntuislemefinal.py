#@markdown Yüz işaret noktalarını görselleştirmek ve yüzü bulanıklaştırmak için bazı işlevler tanımladık. <br/> Aşağıdaki hücreyi çalıştırarak işlevleri etkinleştirin.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2

def yuz_isaretlerini_ciz_ve_bulaniklastir(rgb_goruntu, algilama_sonucu):
    # Algılanan yüzler ve yüzlerdeki işaret noktaları
    yuz_isaret_noktalari_listesi = algilama_sonucu.face_landmarks
    islenmis_goruntu = np.copy(rgb_goruntu)

    # Algılanan yüzleri döngüyle işleme al
    for idx in range(len(yuz_isaret_noktalari_listesi)):
        yuz_isaret_noktalari = yuz_isaret_noktalari_listesi[idx]

        # x, y koordinatlarını CSV ve bulanıklaştırma için al
        x_koordinatlari = [isaret_noktasi.x * rgb_goruntu.shape[1] for isaret_noktasi in yuz_isaret_noktalari]
        y_koordinatlari = [isaret_noktasi.y * rgb_goruntu.shape[0] for isaret_noktasi in yuz_isaret_noktalari]

        # Yüz için sınırlayıcı kutu hesapla
        x_min = int(min(x_koordinatlari)) - 20  # Kenar boşluğu ekle
        x_maks = int(max(x_koordinatlari)) + 20
        y_min = int(min(y_koordinatlari)) - 20
        y_maks = int(max(y_koordinatlari)) + 20

        # Koordinatların görüntü sınırları içinde olduğundan emin ol
        x_min = max(0, x_min)
        x_maks = min(rgb_goruntu.shape[1], x_maks)
        y_min = max(0, y_min)
        y_maks = min(rgb_goruntu.shape[0], y_maks)

        # Yüz bölgesini çıkar
        yuz_bolgesi = islenmis_goruntu[y_min:y_maks, x_min:x_maks]

        # Yüz bölgesine Gaussian bulanıklığı uygula
        bulanik_yuz = cv2.GaussianBlur(yuz_bolgesi, (99, 99), 30)

        # Bulanıklaştırılmış yüzü orijinal görüntüye geri yerleştir
        islenmis_goruntu[y_min:y_maks, x_min:x_maks] = bulanik_yuz

        # İşaret noktası koordinatlarını CSV dosyasına kaydet
        koordinatlar = []
        for isaret_noktasi in yuz_isaret_noktalari:
            koordinatlar.append(str(round(isaret_noktasi.x, 4)))
            koordinatlar.append(str(round(isaret_noktasi.y, 4)))
        
        koordinatlar = ",".join(koordinatlar)
        koordinatlar += f",{etiket}\n"
        with open("veriseti.csv", "a") as dosya:
            dosya.write(koordinatlar)

    return islenmis_goruntu

def yuz_ifadeleri_cubuk_grafik_ciz(yuz_ifadeleri):
    # Yüz ifadelerinin kategori isimlerini ve puanlarını al
    ifade_isimleri = [ifade_kategorisi.category_name for ifade_kategorisi in yuz_ifadeleri]
    ifade_puanlari = [ifade_kategorisi.score for ifade_kategorisi in yuz_ifadeleri]
    ifade_siralari = range(len(ifade_isimleri))

    fig, ax = plt.subplots(figsize=(12, 12))
    cubuk = ax.barh(ifade_siralari, ifade_puanlari, label=[str(x) for x in ifade_siralari])
    ax.set_yticks(ifade_siralari, ifade_isimleri)
    ax.invert_yaxis()

    # Her çubuğa puan etiketi ekle
    for puan, yama in zip(ifade_puanlari, cubuk.patches):
        plt.text(yama.get_x() + yama.get_width(), yama.get_y(), f"{puan:.4f}", va="top")

    ax.set_xlabel('Puan')
    ax.set_title("Yüz İfadeleri")
    plt.tight_layout()
    plt.show()

def sutun_basliklari_olustur():
    with open("veriseti.csv", "w") as dosya:
        satir = ""
        for i in range(1, 479):
            satir = satir + f"x{i},y{i},"
        satir = satir + "Etiket\n"
        dosya.write(satir)

etiket = "mutlu"
# NOT: Aşağıdaki işlev yalnızca ilk etiket için çalışır.
# Diğer etiketler için bu satırı yorum satırı yapın.
sutun_basliklari_olustur()

# ADIM 1: Gerekli modülleri içe aktar
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ADIM 2: Yüz İşaretleyici nesnesi oluştur
temel_ayalar = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
ayarlar = vision.FaceLandmarkerOptions(base_options=temel_ayalar,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
algilayici = vision.FaceLandmarker.create_from_options(ayarlar)

# Web kamerasından görüntü alımı
kamera = cv2.VideoCapture(0)
while kamera.isOpened():
    basari, kare = kamera.read()
    if basari:
        kare = cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)
        mp_goruntu = mp.Image(image_format=mp.ImageFormat.SRGB, data=kare)

        # ADIM 4: Giriş görüntüsünden yüz işaret noktalarını algıla
        algilama_sonucu = algilayici.detect(mp_goruntu)

        # ADIM 5: Algılama sonucunu işleyerek yüzü bulanıklaştır
        islenmis_goruntu = yuz_isaretlerini_ciz_ve_bulaniklastir(mp_goruntu.numpy_view(), algilama_sonucu)
        cv2.imshow("yuz", cv2.cvtColor(islenmis_goruntu, cv2.COLOR_RGB2BGR))
        tus = cv2.waitKey(1)
        if tus == ord('q') or tus == ord('Q'):
            kamera.release()
            cv2.destroyAllWindows()
            break