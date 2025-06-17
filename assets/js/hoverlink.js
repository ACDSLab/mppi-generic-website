// document.addEventListener("DOMContentLoaded", function () {
//   const headers = document.querySelectorAll("h1, h2, h3, h4, h5, h6");

//   headers.forEach((header) => {
//     if (header.id) {
//       const link = document.createElement("a");
//       link.href = `#${header.id}`;
//       link.className = "header-link";
//       link.innerHTML = "🔗"; // You can use an icon here
//       header.appendChild(link);
//     }
//   });
// });

// link.addEventListener("click", function (e) {
//   e.preventDefault();
//   const url = `${window.location.origin}${window.location.pathname}#${header.id}`;
//   navigator.clipboard.writeText(url).then(() => {
//     alert("Link copied to clipboard!");
//   });
// });

document.addEventListener("DOMContentLoaded", function () {
  const headers = document.querySelectorAll("h1, h2, h3, h4, h5, h6");

  headers.forEach((header) => {
    if (header.id) {
      const link = document.createElement("a");
      link.className = "header-link";
      link.innerHTML = "🔗"; // You can replace with an icon or SVG

      // Prevent default link behavior and copy full URL to clipboard
      link.addEventListener("click", function (e) {
        e.preventDefault();
        const url = `${window.location.origin}${window.location.pathname}#${header.id}`;
        navigator.clipboard.writeText(url).then(() => {
          // Optional: show a confirmation
          const originalText = link.innerHTML;
          link.innerHTML = "✅";
          setTimeout(() => {
            link.innerHTML = originalText;
          }, 1000);
        }).catch(err => {
          console.error("Clipboard copy failed:", err);
        });
      });

      header.appendChild(link);
    }
  });
});
