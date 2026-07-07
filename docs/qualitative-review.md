# Qualitative review — 100 user questions

> **What this is:** a point-in-time *manual* review of end-to-end answer quality
> across 100 questions in varied phrasings. It complements the automated,
> reproducible metrics in [evaluation-results.md](evaluation-results.md) by
> judging whole answers (routing, clarification, refusal, hallucination risk) —
> things a retrieval metric cannot capture. It is a human read of a single run,
> not a number that regenerates on every commit; treat the counts as indicative.

Phạm vi: đánh giá dựa trên dữ liệu local trong repo (`data/normalized`, `data/chunks`) và luồng RAG hiện tại. Không xác nhận live với website Vietcombank tại thời điểm đánh giá.

Phương pháp: tạo 100 câu hỏi theo nhiều văn phong, chạy pipeline với GraphRAG và retriever lexical in-memory trên `data/chunks/vietcombank_chunks.jsonl` vì Qdrant local không chạy, sau đó rà thủ công nguồn trả về để đánh giá khả năng trả lời đúng/hợp lý.

Tổng quan:

- Trả lời tốt: 88/100
- Hỏi lại hợp lý hoặc chấp nhận được: 2/100
- Rủi ro/cần sửa: 10/100

| # | Câu hỏi người dùng | Đánh giá | Nhận xét |
|---:|---|---|---|
| 1 | Tài khoản thanh toán Vietcombank có những tiện ích gì? | Tốt | Lấy đúng nguồn Tài khoản thanh toán. |
| 2 | Mở tài khoản số đẹp ở VCB thì có gì khác tài khoản thường? | Tốt | Lấy đúng nguồn Tài khoản số đẹp. |
| 3 | Tài khoản số đẹp phí mở có thấp không, kho số có đa dạng không? | Tốt | Có nguồn đúng, nhưng nên tránh khẳng định phí hiện hành nếu nguồn không có biểu phí chi tiết. |
| 4 | Tôi muốn quản lý tài khoản 24/7 thì dùng sản phẩm nào của Vietcombank? | Tốt | Có thể trả lời từ nhóm tài khoản; nếu muốn tốt hơn nên gợi ý thêm Digibank khi phù hợp. |
| 5 | Cho mình hỏi tài khoản thanh toán Vietcombank miễn phí những gì? | Tốt | Lấy đúng nguồn Tài khoản thanh toán. |
| 6 | tai khoan so dep vcb la gi vay? | Tốt | Câu không dấu vẫn lấy được nguồn Tài khoản số đẹp, dù có thêm vài nguồn nhiễu. |
| 7 | VCB Digibank là dịch vụ gì? | Tốt | Lấy đúng FAQ/Digibank. |
| 8 | Ngân hàng số VCB Digibank có hoạt động 24/7 không? | Tốt | Có nguồn FAQ đúng về hoạt động liên tục 24 giờ. |
| 9 | Tôi đăng ký Digibank rồi thì có thể làm những giao dịch nào? | Rủi ro/cần sửa | Pipeline hỏi lại, trong khi dữ liệu có FAQ đúng. Đây là false clarification. |
| 10 | Làm sao đăng nhập VCB Digibank phiên bản mới? | Tốt | Lấy đúng FAQ đăng nhập phiên bản mới. |
| 11 | Tôi quên mật khẩu VCB Digibank thì xử lý thế nào? | Tốt | Lấy đúng FAQ quên tên đăng nhập/mật khẩu, có một số nguồn phụ về PIN thẻ. |
| 12 | Mật khẩu VCB Digibank nên đặt ra sao và bao lâu phải đổi? | Tốt | Lấy đúng FAQ chính sách mật khẩu. |
| 13 | OTP là gì, VCB có những cách nhận OTP nào? | Rủi ro/cần sửa | Bị từ chối, trong khi đây là câu hỏi thông tin công khai hợp lệ. |
| 14 | Phân biệt SMS OTP và VCB Smart OTP giúp tôi. | Tốt | Lấy đúng FAQ phân biệt phương thức nhận OTP. |
| 15 | SMS Banking của Vietcombank dùng để làm gì? | Tốt | Lấy đúng nguồn SMS Banking. |
| 16 | Phone Banking VCB có hotline nào? | Tốt | Lấy đúng nguồn Phone Banking. |
| 17 | VCB Loyalty là chương trình gì? | Tốt | Lấy đúng nguồn VCB Loyalty. |
| 18 | Điểm VCB Loyalty kiểm tra ở đâu? | Tốt | Có nguồn Loyalty; câu trả lời nên ưu tiên FAQ kiểm tra điểm nếu có. |
| 19 | Sản phẩm tiền gửi nào được tích điểm Loyalty? | Tốt | Lấy đúng FAQ Loyalty về sản phẩm tiền gửi/tiết kiệm được tích điểm. |
| 20 | Tôi ra nước ngoài thì nên nhận OTP kiểu nào để vẫn giao dịch được? | Tốt | Lấy đúng FAQ nhận OTP; không yêu cầu dữ liệu bí mật. |
| 21 | Không nhận được email thông báo kết quả giao dịch trên Digibank thì làm sao? | Tốt | Lấy đúng FAQ liên quan. |
| 22 | Tôi có cần bật cookies trình duyệt khi dùng VCB Digibank không? | Tốt | Lấy đúng FAQ cookies. |
| 23 | Vietcombank hiện có các loại thẻ nào? | Tốt | Trả về đúng nhóm Thẻ tín dụng, Thẻ thanh toán, Dịch vụ thẻ. |
| 24 | Các thẻ tín dụng Vietcombank đang có gồm những thẻ nào? | Tốt | Lấy đúng catalog thẻ tín dụng và các item liên quan. |
| 25 | Có những thẻ thanh toán Vietcombank nào vậy? | Tốt | Lấy đúng catalog thẻ thanh toán. |
| 26 | Vietcombank Vibe Platinum có điểm nổi bật gì? | Tốt | Lấy đúng sản phẩm. |
| 27 | Thẻ Vietcombank Cashplus Platinum American Express hoàn tiền bao nhiêu? | Tốt | Lấy đúng sản phẩm. |
| 28 | Vietcombank Mastercard World có miễn lãi tối đa mấy ngày? | Tốt | Lấy đúng sản phẩm. |
| 29 | Thẻ Vietnam Airlines American Express cộng dặm thế nào? | Tốt | Lấy đúng sản phẩm. |
| 30 | Vietcombank JCB Platinum khác gì JCB thường? | Tốt | Lấy được cả JCB Platinum và JCB thường để so sánh. |
| 31 | Thẻ Vietcombank Visa Snack có ưu đãi hoàn tiền không? | Tốt | Lấy đúng sản phẩm. |
| 32 | Visa Platinum Debit của Vietcombank có rút tiền mặt miễn phí không? | Tốt | Lấy đúng sản phẩm, có thêm nguồn Visa Platinum tín dụng nên cần rerank tốt. |
| 33 | VCB DigiCard là thẻ gì, phí thường niên thế nào? | Tốt | Lấy đúng sản phẩm. |
| 34 | Thẻ Vietcombank Connect24 có hỗ trợ thanh toán không tiếp xúc không? | Tốt | Lấy đúng Connect24 và FAQ không tiếp xúc. |
| 35 | Thẻ Chợ Rẫy Connect24 dùng cho mục đích gì? | Tốt | Lấy đúng sản phẩm. |
| 36 | Thẻ Takashimaya Visa có tích điểm không? | Tốt | Lấy đúng sản phẩm. |
| 37 | Thẻ eVer-link tích điểm VCB Rewards như thế nào? | Tốt | Lấy đúng sản phẩm. |
| 38 | Dịch vụ trả góp linh hoạt trên Digibank là gì? | Tốt | Lấy đúng dịch vụ trả góp linh hoạt. |
| 39 | Thẻ không tiếp xúc của VCB có an toàn không? | Tốt | Lấy đúng FAQ thẻ không tiếp xúc. |
| 40 | Bị nuốt thẻ tại ATM thì tôi phải làm gì? | Tốt | Lấy đúng FAQ bị nuốt thẻ. |
| 41 | Tôi bị mất thẻ thì khóa thẻ qua kênh nào? | Tốt | Có FAQ đúng, nhưng nguồn đầu có thể là mở khóa lại; cần sắp hạng tốt hơn. |
| 42 | Tôi quên mã PIN thẻ thì xin cấp lại như thế nào? | Tốt | Lấy đúng FAQ quên PIN. |
| 43 | Muốn kiểm tra hạn mức còn lại thẻ tín dụng thì xem ở đâu? | Rủi ro/cần sửa | Retrieval ưu tiên catalog thẻ tín dụng hơn FAQ đúng về kiểm tra hạn mức. |
| 44 | Thanh toán sao kê thẻ tín dụng Vietcombank bằng cách nào? | Rủi ro/cần sửa | Retrieval ưu tiên catalog thẻ tín dụng, dễ trả lời lệch thay vì FAQ thanh toán sao kê. |
| 45 | Vietcombank có các gói vay nào hiện nay? | Tốt | Lấy đúng danh mục các gói vay. |
| 46 | Tôi muốn vay mua ô tô, VCB cho vay tối đa bao nhiêu và bao lâu? | Tốt | Lấy đúng sản phẩm Vay mua ô tô. |
| 47 | Điều kiện vay tín chấp theo lương là gì? | Tốt | Lấy đúng sản phẩm. |
| 48 | Vay tiêu dùng có tài sản bảo đảm hạn mức tối đa bao nhiêu? | Tốt | Lấy đúng sản phẩm. |
| 49 | Vay cầm cố giấy tờ có giá ở Vietcombank có gì nổi bật? | Tốt | Lấy đúng sản phẩm. |
| 50 | Các khoản vay nhu cầu bất động sản của VCB gồm gì? | Tốt | Lấy đúng nhóm vay bất động sản. |
| 51 | Nhà Mới Thành Đạt cho vay tối đa bao lâu? | Tốt | Lấy đúng sản phẩm. |
| 52 | Vay mua nhà dự án cần hồ sơ gì? | Tốt | Lấy đúng sản phẩm, có thể trích hồ sơ nếu chunk chứa mục đó. |
| 53 | Vay mua nhà ở, đất ở có thời hạn vay tối đa bao nhiêu năm? | Tốt | Lấy đúng sản phẩm. |
| 54 | Vay xây sửa nhà ở có thể vay tới bao nhiêu phần trăm giá trị xây sửa? | Tốt | Lấy đúng sản phẩm. |
| 55 | An tâm kinh doanh phù hợp đối tượng nào? | Tốt | Lấy đúng sản phẩm. |
| 56 | Kinh doanh tài lộc có thời hạn vay tối đa bao lâu? | Tốt | Lấy đúng sản phẩm. |
| 57 | Vay nâng cấp cơ sở lưu trú du lịch có mức vay và thời hạn ra sao? | Tốt | Lấy đúng sản phẩm, kèm nguồn vay xây mới để so sánh. |
| 58 | Vay xây mới cơ sở lưu trú du lịch khác gì vay nâng cấp? | Tốt | Lấy được cả hai sản phẩm cần so sánh. |
| 59 | Lãi suất vay mua ô tô hiện là bao nhiêu? | Rủi ro/cần thận trọng | Có nguồn sản phẩm nhưng lãi suất hiện hành dễ thay đổi; chatbot chỉ nên nói nếu nguồn có số cụ thể, nếu không phải nêu chưa có. |
| 60 | Tôi kinh doanh nhỏ, muốn vay vốn lưu động thì VCB có sản phẩm nào? | Tốt | Có thể gợi ý An tâm kinh doanh/Kinh doanh tài lộc dựa trên catalog, không nên tư vấn cá nhân hóa. |
| 61 | Vietcombank có những sản phẩm tiết kiệm nào? | Tốt | Lấy đúng danh mục tiết kiệm. |
| 62 | Tiền gửi An Vui có đặc điểm gì? | Tốt | Lấy đúng sản phẩm. |
| 63 | Tiền gửi trực tuyến 2.0 tối thiểu bao nhiêu tiền? | Tốt | Lấy đúng sản phẩm. |
| 64 | Tiền gửi có kỳ hạn trực tuyến giao dịch được online 24/7 không? | Tốt | Lấy đúng sản phẩm. |
| 65 | Tiền gửi tích lũy trực tuyến tối thiểu mỗi lần tích lũy bao nhiêu? | Tốt | Lấy đúng sản phẩm. |
| 66 | Tiền gửi cho con tích lũy định kỳ ra sao? | Tốt | Lấy đúng sản phẩm. |
| 67 | Tiết kiệm tự động tối thiểu bao nhiêu và kỳ hạn tối đa thế nào? | Tốt | Lấy đúng sản phẩm. |
| 68 | Tiền gửi rút gốc linh hoạt có được rút gốc nhiều lần không? | Tốt | Lấy đúng sản phẩm. |
| 69 | Tiết kiệm trả lãi trước hỗ trợ loại tiền nào? | Tốt | Lấy đúng sản phẩm. |
| 70 | Tiết kiệm trả lãi định kỳ yêu cầu số tiền tối thiểu bao nhiêu? | Tốt | Lấy đúng sản phẩm. |
| 71 | Tích lũy kiều hối gửi bằng loại tiền gì? | Tốt | Lấy đúng sản phẩm. |
| 72 | Tiền gửi tiết kiệm trả lãi sau kỳ hạn tối đa bao lâu? | Tốt | Lấy đúng sản phẩm. |
| 73 | Sản phẩm tiết kiệm trực tuyến nào của VCB có thể mở trên Digibank? | Tốt | Lấy đúng nhóm tiết kiệm trực tuyến. |
| 74 | Lãi suất tiền gửi hiện tại của các gói tiết kiệm là bao nhiêu? | Rủi ro/cần thận trọng | Câu hỏi thời điểm; dữ liệu sản phẩm có thể không có bảng lãi suất hiện hành. Cần trả lời thiếu dữ liệu thay vì bịa số. |
| 75 | Vietcombank đang có các gói bảo hiểm FWD nào? | Tốt | Lấy đúng catalog bảo hiểm. |
| 76 | FWD Con vươn xa 2.0 bảo vệ quyền lợi gì? | Tốt | Lấy đúng sản phẩm. |
| 77 | FWD Vững ước mơ là bảo hiểm gì? | Tốt | Lấy đúng sản phẩm, có phân biệt KHCN nếu cần. |
| 78 | FWD Cả nhà vui khỏe quyền lợi lên tới bao nhiêu? | Tốt | Lấy đúng sản phẩm. |
| 79 | FWD Vững ước mơ đóng phí 1 lần có điểm gì đáng chú ý? | Tốt | Lấy đúng biến thể đóng phí 1 lần. |
| 80 | FWD Bảo vệ gia tăng phiên bản trực tuyến tham gia thế nào? | Hỏi lại hợp lý/có thể cải thiện | Pipeline hỏi lại do có nhiều biến thể Bảo vệ gia tăng; câu này khá cụ thể nên có thể cải thiện alias matching. |
| 81 | Bảo hiểm liên kết chung FWD Bảo vệ gia tăng có quyền lợi gì? | Hỏi lại hợp lý | Có biến thể KHCN/KHDN trùng tên; hỏi lại để chọn đối tượng là chấp nhận được. |
| 82 | FWD Bảo hiểm sức khỏe trực tuyến có phí hợp lý không? | Tốt | Lấy đúng sản phẩm. |
| 83 | FWD Bảo hiểm tai nạn trực tuyến có bản cho cá nhân và doanh nghiệp không? | Tốt | Lấy được cả KHCN và KHDN. |
| 84 | Sản phẩm FWD Đầu tư đón đầu thuộc nhóm bảo hiểm nào? | Tốt | Có thể trả lời thuộc Bảo hiểm đầu tư từ catalog. |
| 85 | Tôi muốn bảo hiểm cho cả gia đình thì VCB có gói nào? | Tốt | Có thể gợi ý FWD Cả nhà vui khỏe; cần tránh tư vấn cá nhân hóa. |
| 86 | Tôi muốn hỏi phí bảo hiểm FWD cụ thể từng gói là bao nhiêu? | Rủi ro/cần thận trọng | Câu hỏi phí cụ thể theo từng gói; nếu nguồn thiếu biểu phí phải nói chưa tìm thấy. |
| 87 | Vietcombank có những dịch vụ đầu tư nào? | Tốt | Lấy đúng danh mục đầu tư. |
| 88 | Giao dịch Chứng khoán qua Vietcombank có miễn phí mở tài khoản không? | Tốt | Lấy đúng sản phẩm. |
| 89 | Quỹ mở VCBF đầu tư ban đầu tối thiểu bao nhiêu? | Tốt | Lấy đúng sản phẩm. |
| 90 | Ủy thác quản lý tài khoản của VCB là gì? | Tốt | Lấy đúng sản phẩm. |
| 91 | Chứng chỉ tiền gửi trực tuyến có giao dịch hoàn toàn online không? | Tốt | Lấy đúng sản phẩm. |
| 92 | Dịch vụ báo cáo phân tích của Vietcombank cung cấp gì? | Tốt | Lấy đúng sản phẩm. |
| 93 | Hỗ trợ tài chính trong mục đầu tư là sản phẩm gì? | Tốt | Lấy đúng sản phẩm. |
| 94 | Tôi muốn đầu tư nhưng không biết chọn quỹ nào, chatbot có tư vấn giúp không? | Rủi ro/cần sửa | Đây là tư vấn đầu tư cá nhân hóa. Prompt cấm, nhưng guardrail chưa bắt rõ; nên từ chối/gợi ý tham khảo nguồn chính thức. |
| 95 | VCB có các dịch vụ chuyển và nhận tiền nào? | Tốt | Lấy đúng ba nhóm dịch vụ. |
| 96 | Chuyển và nhận tiền trong nước hỗ trợ kênh giao dịch nào? | Tốt | Lấy đúng sản phẩm. |
| 97 | Nhận kiều hối tại Việt Nam thủ tục có đơn giản không? | Tốt | Lấy đúng sản phẩm và FAQ liên quan. |
| 98 | Chuyển tiền ra nước ngoài qua Vietcombank hỗ trợ bao nhiêu quốc gia? | Tốt | Lấy đúng sản phẩm. |
| 99 | Tôi nhận tiền từ nước ngoài về Việt Nam thì nên dùng dịch vụ nào? | Rủi ro/cần sửa | Bị từ chối, trong khi nên map sang Nhận kiều hối tại Việt Nam. |
| 100 | Biểu phí chuyển tiền ra nước ngoài hiện là bao nhiêu? | Rủi ro/cần thận trọng | Có nguồn sản phẩm nhưng biểu phí hiện hành dễ thay đổi; cần không bịa số và nên yêu cầu kiểm tra kênh chính thức nếu nguồn thiếu. |

Kết luận: chatbot xử lý tốt các câu hỏi có tên sản phẩm/nhóm rõ ràng và các câu hỏi dạng danh mục. Điểm yếu chính nằm ở câu FAQ không có tên sản phẩm cụ thể, câu hỏi có từ khóa nhạy cảm nhưng là thông tin công khai như OTP, một số câu thẻ tín dụng bị catalog lấn FAQ, và các câu hỏi lãi suất/biểu phí/tư vấn cá nhân hóa cần guardrail chặt hơn.
