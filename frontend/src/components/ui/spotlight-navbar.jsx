"use client";

import React, { useEffect, useRef, useState } from "react";
import { animate } from "framer-motion";
import { cn } from "@/lib/utils";

export function SpotlightNavbar({
  items = [
    { label: "Home", href: "#home" },
    { label: "Manifesto", href: "#problem" },
    { label: "Lab", href: "#analyzer" },
    { label: "Batch", href: "#batch" },
    { label: "Strategies", href: "#strategies" },
    { label: "Benchmarks", href: "#benchmarks" },
    { label: "Team", href: "#team" },
  ],
  className,
  onItemClick,
  defaultActiveIndex = 0,
}) {
  const navRef = useRef(null);
  const [activeIndex, setActiveIndex] = useState(defaultActiveIndex);
  const [hoverX, setHoverX] = useState(null);
  const isClickScrollingRef = useRef(false);
  const clickTimeoutRef = useRef(null);

  // Refs for the "light" positions so we can animate them imperatively
  const spotlightX = useRef(0);
  const ambienceX = useRef(0);

  // ScrollSpy to keep active index in sync with user scrolling
  useEffect(() => {
    const handleScroll = () => {
      if (isClickScrollingRef.current) return;
      const windowHeight = window.innerHeight;

      let foundIndex = -1;
      items.forEach((item, idx) => {
        const id = item.href.replace("#", "");
        const el = document.getElementById(id);
        if (el) {
          const rect = el.getBoundingClientRect();
          if (rect.top <= windowHeight * 0.45 && rect.bottom >= windowHeight * 0.15) {
            foundIndex = idx;
          }
        }
      });

      if (foundIndex !== -1 && foundIndex !== activeIndex) {
        setActiveIndex(foundIndex);
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => {
      window.removeEventListener("scroll", handleScroll);
      if (clickTimeoutRef.current) clearTimeout(clickTimeoutRef.current);
    };
  }, [items, activeIndex]);

  // Handle Spotlight cursor tracking
  useEffect(() => {
    if (!navRef.current) return;
    const nav = navRef.current;

    const handleMouseMove = (e) => {
      const rect = nav.getBoundingClientRect();
      const x = e.clientX - rect.left;
      setHoverX(x);
      spotlightX.current = x;
      nav.style.setProperty("--spotlight-x", `${x}px`);
    };

    const handleMouseLeave = () => {
      setHoverX(null);
      // When mouse leaves, spring the spotlight back to the active item
      const activeItem = nav.querySelector(`[data-index="${activeIndex}"]`);
      if (activeItem) {
        const navRect = nav.getBoundingClientRect();
        const itemRect = activeItem.getBoundingClientRect();
        const targetX = itemRect.left - navRect.left + itemRect.width / 2;

        animate(spotlightX.current, targetX, {
          type: "spring",
          stiffness: 240,
          damping: 22,
          onUpdate: (v) => {
            spotlightX.current = v;
            nav.style.setProperty("--spotlight-x", `${v}px`);
          },
        });
      }
    };

    nav.addEventListener("mousemove", handleMouseMove);
    nav.addEventListener("mouseleave", handleMouseLeave);

    return () => {
      nav.removeEventListener("mousemove", handleMouseMove);
      nav.removeEventListener("mouseleave", handleMouseLeave);
    };
  }, [activeIndex]);

  // Handle the "Ambience" (Active Item) Movement
  useEffect(() => {
    if (!navRef.current) return;
    const nav = navRef.current;
    const activeItem = nav.querySelector(`[data-index="${activeIndex}"]`);

    if (activeItem) {
      const navRect = nav.getBoundingClientRect();
      const itemRect = activeItem.getBoundingClientRect();
      const targetX = itemRect.left - navRect.left + itemRect.width / 2;

      animate(ambienceX.current, targetX, {
        type: "spring",
        stiffness: 240,
        damping: 22,
        onUpdate: (v) => {
          ambienceX.current = v;
          nav.style.setProperty("--ambience-x", `${v}px`);
        },
      });
    }
  }, [activeIndex]);

  const handleItemClick = (item, index) => {
    setActiveIndex(index);
    isClickScrollingRef.current = true;
    if (clickTimeoutRef.current) clearTimeout(clickTimeoutRef.current);

    const id = item.href.replace("#", "");
    const target = document.getElementById(id);
    if (target) {
      target.scrollIntoView({ behavior: "smooth" });
    }

    clickTimeoutRef.current = setTimeout(() => {
      isClickScrollingRef.current = false;
    }, 850);

    onItemClick?.(item, index);
  };

  return (
    <div className={cn("relative flex justify-center", className)}>
      <nav
        ref={navRef}
        className={cn(
          "spotlight-nav",
          "relative h-11 rounded-full transition-all duration-300 overflow-hidden"
        )}
        style={{
          backgroundColor: "rgba(10, 14, 22, 0.82)",
          backdropFilter: "blur(20px) saturate(180%)",
          WebkitBackdropFilter: "blur(20px) saturate(180%)",
          border: "1px solid rgba(255, 255, 255, 0.14)",
          boxShadow:
            "0 20px 45px -10px rgba(0, 0, 0, 0.8), 0 0 25px rgba(34, 211, 238, 0.12), inset 0 1px 1px rgba(255, 255, 255, 0.2)",
          "--spotlight-color": "rgba(255, 255, 255, 0.16)",
          "--ambience-color": "rgba(34, 211, 238, 0.95)",
        }}
      >
        {/* Content */}
        <ul className="relative flex items-center h-full px-2 gap-0 z-[10] list-none m-0">
          {items.map((item, idx) => (
            <li key={idx} className="relative h-full flex items-center justify-center">
              <a
                href={item.href}
                data-index={idx}
                onClick={(e) => {
                  e.preventDefault();
                  handleItemClick(item, idx);
                }}
                className={cn(
                  "px-3.5 py-1.5 text-xs sm:text-sm font-medium transition-colors duration-200 rounded-full",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400",
                  activeIndex === idx
                    ? "text-white font-semibold"
                    : "text-slate-400 hover:text-white"
                )}
                style={{
                  textDecoration: "none",
                  userSelect: "none",
                  cursor: "pointer",
                }}
              >
                {item.label}
              </a>
            </li>
          ))}
        </ul>

        {/* LIGHTING LAYERS */}
        {/* 1. The Moving Spotlight (Follows Mouse) */}
        <div
          className="pointer-events-none absolute bottom-0 left-0 w-full h-full z-[1] transition-opacity duration-300"
          style={{
            opacity: hoverX !== null ? 1 : 0,
            background: `
              radial-gradient(
                130px circle at var(--spotlight-x, 0px) 100%, 
                var(--spotlight-color, rgba(255, 255, 255, 0.16)) 0%, 
                transparent 65%
              )
            `,
          }}
        />

        {/* 2. The Active State Ambience (Stays on Active Item) */}
        <div
          className="pointer-events-none absolute bottom-0 left-0 w-full h-[2.5px] z-[2]"
          style={{
            background: `
              radial-gradient(
                70px circle at var(--ambience-x, 0px) 0%, 
                var(--ambience-color, rgba(34, 211, 238, 0.95)) 0%, 
                transparent 100%
              )
            `,
          }}
        />
      </nav>
    </div>
  );
}

export default SpotlightNavbar;
