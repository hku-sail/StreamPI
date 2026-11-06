document.addEventListener('DOMContentLoaded', () => {
  const canAutoplayVideo = (video) => {
    const slide = video.closest('.carousel-slide');
    if (slide && !slide.classList.contains('is-active')) {
      return false;
    }

    return video.dataset.inViewport !== 'false';
  };

  document.querySelectorAll('.task-chart').forEach((chart) => {
    const defaultMetric = chart.dataset.defaultMetric || 'score';
    const buttons = chart.querySelectorAll('.task-toggle-button');
    const panels = chart.querySelectorAll('.task-chart-panel');

    const setMetric = (metric) => {
      buttons.forEach((button) => {
        button.classList.toggle('is-active', button.dataset.metric === metric);
      });

      panels.forEach((panel) => {
        panel.classList.toggle('is-active', panel.dataset.metricPanel === metric);
      });
    };

    buttons.forEach((button) => {
      button.addEventListener('click', () => {
        setMetric(button.dataset.metric);
      });
    });

    setMetric(defaultMetric);
  });

  document.querySelectorAll('[data-carousel]').forEach((carousel) => {
    const track = carousel.querySelector('.carousel-track');
    const slides = Array.from(carousel.querySelectorAll('.carousel-slide'));
    const buttons = Array.from(carousel.querySelectorAll('.carousel-dot'));
    let currentIndex = 0;
    let startX = 0;
    let currentTranslate = 0;
    let isDragging = false;

    if (!track || slides.length < 2) {
      return;
    }

    const syncSlidePlayback = (activeIndex, reset = false) => {
      slides.forEach((slide, index) => {
        slide.classList.toggle('is-active', index === activeIndex);
        slide.querySelectorAll('video').forEach((video) => {
          if (index === activeIndex && canAutoplayVideo(video)) {
            if (reset) {
              video.currentTime = 0;
            }
            video.play().catch(() => {});
          } else {
            video.pause();
          }
        });
      });
    };

    const setSlide = (index, reset = true) => {
      currentIndex = Math.max(0, Math.min(index, slides.length - 1));
      track.style.transform = `translateX(-${currentIndex * 100}%)`;
      buttons.forEach((button) => {
        button.classList.toggle('is-active', Number(button.dataset.slide) === currentIndex);
      });
      syncSlidePlayback(currentIndex, reset);
    };

    buttons.forEach((button) => {
      button.addEventListener('click', () => {
        setSlide(Number(button.dataset.slide), true);
      });
    });

    const onPointerDown = (clientX) => {
      startX = clientX;
      currentTranslate = -currentIndex * carousel.offsetWidth;
      isDragging = true;
      track.style.transition = 'none';
    };

    const onPointerMove = (clientX) => {
      if (!isDragging) {
        return;
      }
      const deltaX = clientX - startX;
      track.style.transform = `translateX(${currentTranslate + deltaX}px)`;
    };

    const onPointerUp = (clientX) => {
      if (!isDragging) {
        return;
      }
      const deltaX = clientX - startX;
      isDragging = false;
      track.style.transition = '';

      if (Math.abs(deltaX) > 60) {
        if (deltaX < 0) {
          setSlide(currentIndex + 1, true);
        } else {
          setSlide(currentIndex - 1, true);
        }
      } else {
        setSlide(currentIndex, false);
      }
    };

    track.addEventListener('mousedown', (event) => {
      onPointerDown(event.clientX);
    });

    window.addEventListener('mousemove', (event) => {
      onPointerMove(event.clientX);
    });

    window.addEventListener('mouseup', (event) => {
      onPointerUp(event.clientX);
    });

    track.addEventListener('mouseleave', () => {
      if (isDragging) {
        onPointerUp(startX);
      }
    });

    track.addEventListener('touchstart', (event) => {
      onPointerDown(event.touches[0].clientX);
    }, { passive: true });

    track.addEventListener('touchmove', (event) => {
      onPointerMove(event.touches[0].clientX);
    }, { passive: true });

    track.addEventListener('touchend', (event) => {
      const touch = event.changedTouches[0];
      onPointerUp(touch ? touch.clientX : startX);
    });

    setSlide(0, true);
  });

  const playbackObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const video = entry.target;
      const isVisible = entry.isIntersecting;
      video.dataset.inViewport = isVisible ? 'true' : 'false';

      if (isVisible && canAutoplayVideo(video)) {
        video.play().catch(() => {});
      } else {
        video.pause();
      }
    });
  }, {
    threshold: 0.35,
  });

  document.querySelectorAll('video').forEach((video) => {
    video.dataset.inViewport = 'false';
    playbackObserver.observe(video);
  });
});
