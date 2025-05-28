# 1. Mô tả bài toán

Nhận diện cảm xúc khuôn mặt là một bài toán trong lĩnh vực **thị giác máy tính (Computer Vision)** và **học máy (Machine Learning)**. Mục tiêu của bài toán là xác định cảm xúc của một người dựa trên hình ảnh hoặc video khuôn mặt (trong đồ án lần này sẽ tập trung vào xác định cảm xúc dựa trên hình ảnh). Đây là một bài toán có nhiều ứng dụng thực tiễn trong AI, y tế, marketing và tương tác người - máy.

## Ứng dụng của Nhận diện cảm xúc khuôn mặt

- **Giáo dục:** Phản ứng của người học trong thời gian thực và sự tham gia vào nội dung là một thước đo lường cho hiệu quả của bài giảng.
- **Tiếp thị:** Đây là một cách tuyệt vời để các công ty kinh doanh phân tích cách khách hàng phản hồi với quảng cáo, sản phẩm, bao bì và thiết kế cửa hàng của họ.
- **Chơi game:** Với sự ra đời của game thực tế ảo gần với trải nghiệm thực tế, nhận dạng cảm xúc khuôn mặt đóng một vai trò quan trọng để cải thiện trải nghiệm chơi trò chơi.
- **Bảo mật:** Nó có thể giúp xác định hành vi đáng ngờ trong đám đông và có thể được sử dụng để ngăn chặn tội phạm và những kẻ khủng bố tiềm năng.
- **Chăm sóc sức khỏe:** Nó có thể hữu ích trong việc tự động hóa dịch vụ y tế. Cả sức khỏe thể chất và tinh thần có thể được phân tích thông qua ứng dụng này.



# 2. Các tiêu chí đánh giá độ hiệu quả của mô hình

Khi đánh giá một mô hình phân loại, chúng ta thường sử dụng **ma trận nhầm lẫn (Confusion Matrix)** để xác định các chỉ số đánh giá. Giả sử mô hình phân loại nhị phân (có hai lớp: Dương tính và Âm tính), ma trận nhầm lẫn có dạng:

|                        | Dự đoán Dương tính | Dự đoán Âm tính   |
|------------------------|--------------------|--------------------|
| Thực tế Dương tính (P) | True Positive (TP) | False Negative (FN)|
| Thực tế Âm tính (N)    | False Positive (FP)| True Negative (TN) |

**Trong đó:**
- **True Positive (TP):** Số lượng mẫu thực sự dương tính và được mô hình dự đoán đúng.
- **False Positive (FP):** Số lượng mẫu thực sự âm tính nhưng bị mô hình dự đoán sai thành dương tính *(lỗi Type I)*.
- **False Negative (FN):** Số lượng mẫu thực sự dương tính nhưng bị mô hình dự đoán sai thành âm tính *(lỗi Type II)*.
- **True Negative (TN):** Số lượng mẫu thực sự âm tính và được mô hình dự đoán đúng.

---

## 2.1. Classification Accuracy

Accuracy = (TP + TN) / (TP + TN + FP + FN)

- **Accuracy** đo lường tỷ lệ dự đoán đúng trên tổng số mẫu.
- Đây là chỉ số đơn giản và dễ hiểu, phản ánh tổng thể hiệu suất mô hình.
- Tuy nhiên, Accuracy có thể gây hiểu lầm trong các bài toán có **dữ liệu mất cân đối (imbalanced dataset)**. Ví dụ, nếu 95% dữ liệu thuộc một lớp và mô hình luôn dự đoán lớp đó, Accuracy vẫn có thể cao nhưng mô hình không thực sự hiệu quả.

---

## 2.2. Precision
Precision = TP / (TP + FP)


- **Precision** đo lường tỷ lệ dự đoán đúng trong tất cả các dự đoán dương tính.
- Precision cao đồng nghĩa với việc mô hình **hiếm khi dự đoán sai dương tính**.
- Thường quan trọng khi **False Positive có ảnh hưởng lớn** (VD: chẩn đoán nhầm bệnh).

---

## 2.3. Recall

Recall = TP / (TP + FN)

- **Recall** đo lường khả năng mô hình phát hiện đúng tất cả các trường hợp dương tính.
- Quan trọng khi **False Negative có hậu quả nghiêm trọng** (VD: bỏ sót ung thư).

---

## 2.4. F1-score

F1-score = 2 × (Precision × Recall) / (Precision + Recall)

- **F1-score** là trung bình điều hòa giữa Precision và Recall.
- Dùng khi muốn **cân bằng giữa Precision và Recall**, đặc biệt trong bài toán có dữ liệu **mất cân bằng**.
- Nếu một chỉ số cao nhưng chỉ số còn lại thấp, F1-score sẽ phản ánh điều đó.

# 3. Mô hình sử dụng

Đây sẽ là kiến trúc được sử dụng trong đồ án này. Đồ án sẽ không sử dụng **transfer learning** mà sẽ xây dựng một mạng **DCNN cơ bản** để giải quyết bài toán nhận diện cảm xúc khuôn mặt.

DCNN là một loại mạng nơ-ron tích chập sâu, được sử dụng rộng rãi trong lĩnh vực thị giác máy tính để xử lý và phân tích hình ảnh. DCNN là phiên bản mở rộng của mạng CNN (Convolutional Neural Network), với nhiều lớp hơn nhằm tăng cường khả năng trích xuất đặc trưng và học biểu diễn phức tạp.

Mạng này bao gồm nhiều lớp tích chập, trong đó mỗi lớp sử dụng các bộ lọc (kernels) để phát hiện các đặc trưng như cạnh, góc, kết cấu và các đặc trưng cấp cao hơn. Sau mỗi lớp tích chập thường có một hàm kích hoạt phi tuyến, phổ biến nhất là **ReLU**, giúp mô hình học được các đặc trưng phức tạp và cải thiện khả năng phân biệt giữa các đối tượng.

Ngoài ra, các lớp **pooling**, như Max Pooling, được sử dụng để giảm kích thước dữ liệu, tối ưu hóa hiệu suất tính toán và giảm hiện tượng quá khớp (overfitting). Cuối cùng, các lớp kết nối đầy đủ (**Fully Connected**) giúp chuyển đổi đặc trưng thành đầu ra phù hợp, hỗ trợ việc phân loại hoặc nhận diện.

Nhờ có độ sâu lớn với nhiều lớp, DCNN có khả năng học các biểu diễn phức tạp và đạt hiệu suất cao trong các tác vụ như phân loại ảnh, nhận diện đối tượng và xử lý hình ảnh y tế.

# 4. Phương pháp đề xuất

## 4.1 Xây dựng Dataset

- **Thư mục `data`:** Gồm 4 thư mục con mỗi thư mục sẽ có một chức năng riêng.
  
  + **Thư mục `dataset_full`:** Là tập FER-2013 bao gồm ~30000 ảnh bao gồm 7 cảm xúc của bài toán chia làm 2 thư mục `train` và `test`, mỗi thư mục sẽ có 7 thư mục con là nhãn của 7 biểu cảm.
    
  + **Thư mục `dataset_split:`** Là thư mục `dataset_full` chia lại làm 3 tập gồm `train` , `test` và `val` với tỉ lệ tương ứng là 70% , 15% , 15%. (tập `test` được sử dụng để cho ra kết quả test trên tập FER_2013)
    
  + **Thư mục `test_data:`** Là tập ảnh tự thu thập dùng để tiền xử lí và test mô hình. (tập `test_data` được sử dụng để đưa ra kết quả test dựa trên hình ảnh thu thập được)
    
  + **Thư mục `captured_images:`** Ảnh sau khi được dự đoán từ `emotion_capture` sẽ được lưu tại đây.

## 4.2. Tiền xử lí hình ảnh

Ta sẽ tiền sử lí dữ liệu thu thập được để đưa vào test với mô hình đã xây dựng.

### 4.2.1. Nhận diện khuôn mặt bằng Haar Cascade
#### Các bước xử lý:

- Sử dụng phương thức `detectMultiScale()`, thuật toán sẽ quét ảnh ở nhiều tỷ lệ khác nhau để xác định vị trí khuôn mặt.
- Một số tham số quan trọng:
  - `scaleFactor=1.1`: giảm kích thước ảnh 10% sau mỗi lần quét để phát hiện khuôn mặt ở nhiều kích cỡ khác nhau.
  - `minNeighbors=5`: số lượng vùng lân cận tối thiểu cần xác nhận để quyết định có phải khuôn mặt thật hay không (giúp giảm false positive).
  - `minSize=(30, 30)`: kích thước tối thiểu của khuôn mặt cần phát hiện.

Khi khuôn mặt được nhận diện, **OpenCV** sẽ vẽ **hình chữ nhật** xung quanh khuôn mặt và hiển thị kết quả đầu ra.

### 4.2.2. Cân bằng sáng và độ tương phản bằng phương pháp Histogram Equalization và CLAHE

####  Histogram Equalization

Phương pháp **Histogram Equalization** được sử dụng để phân bố lại mức độ sáng trên toàn bộ ảnh, giúp làm rõ các chi tiết trong vùng tối hoặc sáng quá mức. OpenCV cung cấp hàm `cv2.equalizeHist()` để thực hiện việc này một cách hiệu quả.

>  Tuy nhiên, Histogram Equalization có thể gây mất chi tiết nếu độ sáng thay đổi quá mạnh trên ảnh.

####  CLAHE (Contrast Limited Adaptive Histogram Equalization)

Để khắc phục nhược điểm của Histogram Equalization, phương pháp **CLAHE** được áp dụng. CLAHE chia ảnh thành các vùng nhỏ và thực hiện cân bằng sáng riêng cho từng vùng, giúp giữ lại chi tiết ở những khu vực có độ tương phản thấp.

- **Các tham số quan trọng:**
  - `clipLimit=2.0`: giới hạn điều chỉnh độ tương phản để tránh làm sáng quá mức.
  - `tileGridSize=(8, 8)`: số lượng vùng chia nhỏ trong ảnh.

Khi áp dụng CLAHE, ảnh đầu vào trở nên rõ ràng hơn, giúp mô hình nhận diện khuôn mặt chính xác hơn trong nhiều điều kiện ánh sáng khác nhau.

#### Resize và chuẩn hóa dữ liệu

Trong quá trình tiền xử lý ảnh, việc chuẩn hóa kích thước và giá trị pixel là bước thiết yếu để đảm bảo dữ liệu đầu vào đồng nhất và phù hợp với mô hình học sâu.

####  Thay đổi kích thước ảnh

Sử dụng phương thức `cv2.resize()`, thuật toán sẽ thay đổi kích thước ảnh về một giá trị cố định, phổ biến là **48x48** pixel. Việc này giúp:
- Đảm bảo tính đồng nhất giữa các ảnh đầu vào.
- Giảm độ phức tạp tính toán.
- Giúp mạng CNN dễ dàng trích xuất đặc trưng mà không bị ảnh hưởng bởi sự chênh lệch kích thước ảnh.

#### Chuẩn hóa giá trị pixel

Sau khi resize, dữ liệu được chuẩn hóa bằng cách chia toàn bộ giá trị pixel cho 255, đưa các giá trị về khoảng **[0, 1]**. Phương pháp này giúp:
- Giảm sự chênh lệch độ sáng giữa các ảnh.
- Giúp mô hình hội tụ nhanh hơn trong quá trình huấn luyện.

Khi ảnh đã được resize và chuẩn hóa, nó có thể được đưa vào mô hình **CNN** để thực hiện **nhận diện cảm xúc** một cách hiệu quả.

## Kiến trúc mạng DCNN

Đầu vào là tập `train` và `val` thuộc thư mục `dataset_split` trong đó :
###  Tập `train`
- Dùng để **huấn luyện mô hình**, giúp mô hình học các đặc trưng từ ảnh và tối ưu các tham số (weights).
- Các nhãn cảm xúc tương ứng giúp mô hình học cách phân loại ảnh chính xác.

###  Tập `val`
- Dùng để **đánh giá hiệu suất mô hình trong quá trình huấn luyện**.
- Giúp:
  - Phát hiện **overfitting** (mô hình học quá kỹ trên dữ liệu huấn luyện).
  - Điều chỉnh các **hyperparameters** như số epoch, learning rate, số lớp,...

