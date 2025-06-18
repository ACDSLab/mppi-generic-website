// Add a permalink button that copies URLs pointing to headers
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
          // link.innerHTML = "✅";
          link.innerHTML = "Copied!";
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

// Copy code block to clipboard
document.addEventListener("DOMContentLoaded", () => {
  // Handle <figure.highlight> and <div.highlight>
  const containerBlocks = document.querySelectorAll("figure.highlight, div.highlight");

  containerBlocks.forEach((block) => {
    if (block.querySelector(".copy-button")) return;

    let codeBlock;

    const tdCode = block.querySelector("td.code");
    if (tdCode) {
      codeBlock = tdCode;
    } else {
      codeBlock = block.querySelector("pre");
    }

    if (!codeBlock) return;

    addCopyButton(block, codeBlock);
  });

  // Handle standalone <code class="language-BibTex"> (outside of .highlight containers)
  const bibtexBlocks = document.querySelectorAll("code.language-BibTex");

  bibtexBlocks.forEach((code) => {
    // Avoid duplicates if it's already inside a .highlight block
    if (code.closest(".highlight") || code.parentElement.querySelector(".copy-button")) return;

    // Wrap in a container for relative positioning
    const wrapper = document.createElement("div");
    wrapper.className = "code-container";
    code.parentElement.insertBefore(wrapper, code);
    wrapper.appendChild(code);

    addCopyButton(wrapper, code);
  });

  function addCopyButton(container, codeElement) {
    const button = document.createElement("button");
    button.className = "copy-button";
    button.type = "button";
    button.innerText = "Copy";

    button.addEventListener("click", () => {
      const code = codeElement.innerText;
      navigator.clipboard.writeText(code).then(() => {
        button.innerText = "Copied!";
        setTimeout(() => (button.innerText = "Copy"), 2000);
      }).catch((err) => {
        console.error("Copy failed", err);
        button.innerText = "Error";
      });
    });

    container.classList.add("code-container");
    container.appendChild(button);
  }
});
