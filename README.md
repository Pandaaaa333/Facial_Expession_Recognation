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

