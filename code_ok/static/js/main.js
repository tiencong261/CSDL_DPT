// JavaScript cho ứng dụng tìm kiếm ảnh tương tự

document.addEventListener("DOMContentLoaded", function () {
  // Xử lý hiển thị tên file khi chọn file
  const fileInput = document.getElementById("file");
  if (fileInput) {
    fileInput.addEventListener("change", function () {
      const fileName = this.files[0]
        ? this.files[0].name
        : "Không có file nào được chọn";
      const fileLabel = document.querySelector('.form-label[for="file"]');
      fileLabel.textContent = "File đã chọn: " + fileName;
    });
  }

  // Tự động đóng thông báo alert sau 5 giây
  const alerts = document.querySelectorAll(".alert");
  alerts.forEach(function (alert) {
    setTimeout(function () {
      const closeButton = alert.querySelector(".btn-close");
      if (closeButton) {
        closeButton.click();
      }
    }, 5000);
  });

  // Kiểm tra form trước khi submit
  const forms = document.querySelectorAll("form");
  forms.forEach(function (form) {
    form.addEventListener("submit", function (event) {
      let isValid = true;

      // Kiểm tra form upload file
      if (form.getAttribute("enctype") === "multipart/form-data") {
        const fileInput = form.querySelector('input[type="file"]');
        if (fileInput && fileInput.files.length === 0) {
          alert("Vui lòng chọn một file ảnh để tìm kiếm.");
          isValid = false;
        }
      }
      // Kiểm tra form URL
      else {
        const urlInput = form.querySelector('input[type="url"]');
        if (urlInput && urlInput.value.trim() === "") {
          alert("Vui lòng nhập URL của ảnh để tìm kiếm.");
          isValid = false;
        }
      }

      if (!isValid) {
        event.preventDefault();
      } else {
        // Hiển thị thông báo đang xử lý
        const submitButton = form.querySelector('button[type="submit"]');
        if (submitButton) {
          submitButton.disabled = true;
          submitButton.innerHTML =
            '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Đang xử lý...';
        }
      }
    });
  });
});
