import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# --- CẤU HÌNH GIAO DIỆN WEB ---
st.set_page_config(
    page_title="Mô phỏng LTE/5G ",
    layout="wide",
)

st.title(
    " Web Demo Đồ án: Mô phỏng và So sánh Kỹ thuật Điều chế QPSK, 16-QAM,"
    " 64-QAM, 256-QAM trong LTE/5G"
)
st.write(
    "Hệ thống tích hợp toàn bộ module: Phát & Ánh xạ  -> Kênh truyền & Fading"
    " -> Thu, Giải điều chế Euclidean & Đo BER/SER "
)

# ==========================================
# 1. KHỐI PHÁT & ÁNH XẠ CHÒM SAO
# ==========================================


def tao_chom_sao_va_map(M):
  if M == 4:
    # QPSK khớp chuẩn với trọng số nhị phân của hàm dieu_che_bits
    bits_map = [
        np.array([0, 0]),  # Index 0
        np.array([0, 1]),  # Index 1
        np.array([1, 0]),  # Index 2
        np.array([1, 1]),  # Index 3
    ]
    points = np.array(
        [
            (1 + 1j) / np.sqrt(2),  # Tương ứng [0, 0] (Góc phần tư 1)
            (1 - 1j) / np.sqrt(2),  # Tương ứng [0, 1] (Góc phần tư 4)
            (-1 + 1j) / np.sqrt(2),  # Tương ứng [1, 0] (Góc phần tư 2)
            (-1 - 1j) / np.sqrt(2),  # Tương ứng [1, 1] (Góc phần tư 3)
        ]
    )
    return points, bits_map

  # Các bậc QAM cao hơn giữ nguyên như cũ
  m = int(np.sqrt(M))
  k = int(np.log2(M))
  muc = np.arange(-(m - 1), m, 2)
  muc = muc / np.sqrt(2 * (M - 1) / 3)
  I, Q = np.meshgrid(muc, muc)
  points = I.flatten() + 1j * Q.flatten()

  bits_map = []
  for i in range(M):
    bits_val = [(i >> bit_idx) & 1 for bit_idx in range(k - 1, -1, -1)]
    bits_map.append(np.array(bits_val))

  return points, bits_map


def dieu_che_bits(bits, M):
  k = int(np.log2(M))
  remainder = len(bits) % k
  if remainder != 0:
    bits = np.pad(bits, (0, k - remainder), "constant")
  bits_reshaped = bits.reshape((-1, k))

  powers = 2 ** np.arange(k - 1, -1, -1)
  symbol_indices = np.dot(bits_reshaped, powers)

  points, _ = tao_chom_sao_va_map(M)
  return symbol_indices, points[symbol_indices]


# ==========================================
# 2. KHỐI KÊNH TRUYỀN
# ==========================================


def kenh_truyen(tx_symbols, snr_db, M, channel_type):
  k = int(np.log2(M))
  snr_lin = 10 ** (snr_db / 10.0)
  noise_var = 1 / (k * snr_lin)

  noise = np.sqrt(noise_var / 2) * (
      np.random.randn(len(tx_symbols)) + 1j * np.random.randn(len(tx_symbols))
  )

  if channel_type == "AWGN":
    return tx_symbols + noise
  elif channel_type == "Rayleigh Fading":
    h = (
        np.random.randn(len(tx_symbols))
        + 1j * np.random.randn(len(tx_symbols))
    ) / np.sqrt(2)
    rx = h * tx_symbols + noise
    return rx / h  # Cân bằng kênh tại phía thu (Equalization)
  return tx_symbols + noise


# ==========================================
# 3. KHỐI THU, GIẢI ĐIỀU CHẾ & ĐO ĐẠC
# ==========================================


def qam_demodulate(received_symbols, constellation_points, bit_mapping):
  demod_bits = []
  # Chia nhỏ block để tối ưu RAM khi tính khoảng cách Euclidean
  for start in range(0, len(received_symbols), 2000):
    end = min(start + 2000, len(received_symbols))
    distances = (
        np.abs(received_symbols[start:end, None] - constellation_points[None, :])
        ** 2
    )
    min_indices = np.argmin(distances, axis=1)
    for idx in min_indices:
      demod_bits.extend(bit_mapping[idx])
  return np.array(demod_bits, dtype=int)


def calculate_error_rates(tx_bits, rx_bits, tx_symbols, rx_detected_symbols, M):
  min_len_bits = min(len(tx_bits), len(rx_bits))
  bit_errors = np.sum(tx_bits[:min_len_bits] != rx_bits[:min_len_bits])
  ber = bit_errors / min_len_bits

  min_len_syms = min(len(tx_symbols), len(rx_detected_symbols))
  symbol_errors = np.sum(
      tx_symbols[:min_len_syms] != rx_detected_symbols[:min_len_syms]
  )
  ser = symbol_errors / min_len_syms

  return ber, ser


# ==========================================
# THANH ĐIỀU KHIỂN (SIDEBAR) TRÊN WEB
# ==========================================
st.sidebar.header(" Cấu hình thông số mô phỏng")
selected_m = st.sidebar.selectbox(
    "Chọn Kỹ thuật Điều chế",
    [4, 16, 64, 256],
    format_func=lambda x: f"{x}-QAM" if x > 4 else "QPSK",
)
selected_channel = st.sidebar.selectbox(
    "Chọn Kênh Truyền", ["AWGN", "Rayleigh Fading"]
)

# Thanh trượt SNR chuẩn từ -5 đến 30 dB
snr_db_slider = st.sidebar.slider(
    "Chọn mức SNR hiện tại (dB)", min_value=-5, max_value=30, value=15, step=1
)
N_bits = st.sidebar.selectbox(
    "Số lượng bit mô phỏng", [10000, 20000, 30000, 40000, 50000], index=1
)

# Gợi ý Thuật toán AMC tự động
st.sidebar.subheader(" Gợi ý Thuật toán AMC")
if snr_db_slider < 5:
  st.sidebar.warning(
      "SNR Thấp (<5 dB): Khuyên dùng **QPSK** để chống nhiễu tối đa."
  )
elif snr_db_slider < 15:
  st.sidebar.info(
      "SNR Trung bình (5-15 dB): Khuyên dùng **16-QAM hoặc 64-QAM**."
  )
else:
  st.sidebar.success(
      "SNR Cao (>15 dB): Khuyên dùng **256-QAM** để đạt tốc độ truyền tải tối"
      " đa trong 5G."
  )

run_button = st.sidebar.button("Chạy mô phỏng hệ thống", key="run_sim_btn")

# ==========================================
# KHU VỰC HIỂN THỊ CHÍNH (DASHBOARD)
# ==========================================
if run_button:
  mod_name = f"{selected_m}-QAM" if selected_m > 4 else "QPSK"
  with st.spinner(
      f"Đang thực hiện quét dải SNR từ -5 đến 30 dB cho {mod_name} qua kênh"
      f" {selected_channel}..."
  ):
    # Vòng lặp quét dải SNR từ -5 dB đến 30 dB
    snr_db_range = np.arange(-5, 31, 2)

    # Khởi tạo dữ liệu phát từ TV1
    tx_bits = np.random.randint(0, 2, N_bits)
    symbol_indices_tx, tx_symbols = dieu_che_bits(tx_bits, selected_m)
    points, bit_mapping = tao_chom_sao_va_map(selected_m)

    ber_list = []
    ser_list = []

    for snr in snr_db_range:
      # Qua kênh truyền của TV2
      rx_symbols = kenh_truyen(tx_symbols, snr, selected_m, selected_channel)

      # Giải điều chế Euclidean và đo đạc của
      rx_bits = qam_demodulate(rx_symbols, points, bit_mapping)
      chom_sao_diem, _ = tao_chom_sao_va_map(selected_m)
      distances_rx = (
          np.abs(rx_symbols[:, None] - chom_sao_diem[None, :]) ** 2
      )
      symbol_indices_rx = np.argmin(distances_rx, axis=1)

      ber, ser = calculate_error_rates(
          tx_bits, rx_bits, symbol_indices_tx, symbol_indices_rx, selected_m
      )
      ber_list.append(ber)
      ser_list.append(ser)

    # Chia giao diện thành 2 cột hiển thị
    col1, col2 = st.columns(2)

    with col1:
      st.subheader(f"Đường cong BER & SER theo dải SNR (-5 đến 30 dB)")
      fig, ax = plt.subplots(figsize=(6, 4))
      ax.semilogy(snr_db_range, ber_list, "b-o", linewidth=2, label="BER (Sim)")
      ax.semilogy(snr_db_range, ser_list, "r--s", linewidth=2, label="SER (Sim)")
      ax.grid(True, which="both", linestyle="--", alpha=0.7)
      ax.set_xlabel("SNR (dB)")
      ax.set_ylabel("Tỷ lệ lỗi (Log scale)")
      ax.set_title(f"Hiệu năng {mod_name} qua kênh {selected_channel}")
      ax.legend()
      st.pyplot(fig)

    with col2:
      st.subheader(
          f"Sơ đồ Chòm sao (Constellation) tại SNR = {snr_db_slider} dB"
      )
      fig2, ax2 = plt.subplots(figsize=(6, 4))
      sample_rx = kenh_truyen(
          tx_symbols[:300], snr_db_slider, selected_m, selected_channel
      )
      ax2.scatter(sample_rx.real, sample_rx.imag, s=12, color="purple", alpha=0.7)
      ax2.grid(True)
      ax2.set_xlabel("Phần thực (I)")
      ax2.set_ylabel("Phần ảo (Q)")
      ax2.set_aspect("equal")
      st.pyplot(fig2)

    st.success(" Mô phỏng hoàn tất toàn bộ hệ thống ")
else:
  st.info(
      " Hãy cấu hình thông số ở thanh bên trái và bấm **'Chạy mô phỏng hệ"
      " thống'** để chạy toàn bộ thuật toán của nhóm."
  )
