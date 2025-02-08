<script>
    document.querySelector(".hamburger").addEventListener("click", function() {
        document.querySelector(".nav-links").classList.toggle("active")
    });
</script>


document.addEventListener("DOMContentLoaded", function () {
    // Smooth scrolling for all navbar links
    document.querySelectorAll('.nav-link, .dropdown-item').forEach(link => {
        link.addEventListener("click", function (event) {
            if (this.getAttribute("href").startsWith("#")) {
                event.preventDefault();
                const targetId = this.getAttribute("href").substring(1);
                const targetElement = document.getElementById(targetId);
                
                if (targetElement) {
                    window.scrollTo({
                        top: targetElement.offsetTop - 50, // Adjust for fixed navbar
                        behavior: "smooth"
                    });
                }
            }
        });
    });

    // Close the navbar when clicking a link (for mobile)
    document.querySelectorAll('.nav-link, .dropdown-item').forEach(link => {
        link.addEventListener("click", function () {
            document.querySelector(".navbar-collapse").classList.remove("show");
        });
    });
});

document.addEventListener("DOMContentLoaded", function () {
    // Smooth Scroll for "Explore More" button
    document.querySelector(".btn-success").addEventListener("click", function (e) {
        e.preventDefault();
        document.getElementById("about").scrollIntoView({ behavior: "smooth" });
    });

    // Fade-in effect for About Section
    const aboutSection = document.getElementById("about");
    function fadeInOnScroll() {
        const rect = aboutSection.getBoundingClientRect();
        const windowHeight = window.innerHeight;
        if (rect.top < windowHeight - 100) {
            aboutSection.classList.add("fade-in");
        }
    }
    
    // Listen for scroll event
    window.addEventListener("scroll", fadeInOnScroll);
    fadeInOnScroll(); // Run on page load
});