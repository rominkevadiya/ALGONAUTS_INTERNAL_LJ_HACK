import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Home,
  BookOpen,
  ScanLine,
  Layers,
  Eye,
  BarChart3,
  Users,
} from "lucide-react";

// Ordered to strictly match website page flow: Home -> Manifesto -> Lab -> Strategies -> 3D Visuals -> Benchmarks -> Team
const NAV_ITEMS = [
  { id: "home", label: "Home", href: "#home", icon: Home },
  { id: "problem", label: "Manifesto", href: "#problem", icon: BookOpen },
  { id: "analyzer", label: "Lab", href: "#analyzer", icon: ScanLine },
  { id: "strategies", label: "Strategies", href: "#strategies", icon: Layers },
  { id: "visuals", label: "3D Visuals", href: "#visuals", icon: Eye },
  { id: "benchmarks", label: "Benchmarks", href: "#benchmarks", icon: BarChart3 },
  { id: "team", label: "Team", href: "#team", icon: Users },
];

export function LiquidGlassNavbar() {
  const [activeTab, setActiveTab] = useState("home");
  const [hoveredTab, setHoveredTab] = useState(null);
  const isClickScrollingRef = useRef(false);
  const clickTimeoutRef = useRef(null);

  // ScrollSpy to update active tab based on scroll position
  useEffect(() => {
    const handleScroll = () => {
      // If user recently clicked a tab, lock active indicator to target tab and ignore intermediate scroll events
      if (isClickScrollingRef.current) return;

      const windowHeight = window.innerHeight;

      // Find current section in view
      let currentSection = "home";
      for (const item of NAV_ITEMS) {
        const el = document.getElementById(item.id);
        if (el) {
          const rect = el.getBoundingClientRect();
          // Check if section occupies the central viewing area
          if (rect.top <= windowHeight * 0.45 && rect.bottom >= windowHeight * 0.15) {
            currentSection = item.id;
          }
        }
      }
      setActiveTab(currentSection);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => {
      window.removeEventListener("scroll", handleScroll);
      if (clickTimeoutRef.current) clearTimeout(clickTimeoutRef.current);
    };
  }, []);

  const handleClick = (e, item) => {
    e.preventDefault();
    setActiveTab(item.id);

    // Lock ScrollSpy while programmatic smooth scroll executes
    isClickScrollingRef.current = true;
    if (clickTimeoutRef.current) clearTimeout(clickTimeoutRef.current);

    const target = document.getElementById(item.id);
    if (target) {
      target.scrollIntoView({ behavior: "smooth" });
    }

    // Release lock once smooth scrolling completes
    clickTimeoutRef.current = setTimeout(() => {
      isClickScrollingRef.current = false;
    }, 850);
  };

  return (
    <div
      className="liquid-nav-container"
      style={{
        position: "fixed",
        bottom: "28px",
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 9999,
        maxWidth: "96vw",
        pointerEvents: "auto",
      }}
    >
      {/* Outer translucent glass dock capsule (iOS-26 inspired) */}
      <nav
        className="liquid-nav-bar"
        style={{
          position: "relative",
          display: "flex",
          alignItems: "center",
          gap: "3px",
          padding: "5px 8px",
          height: "52px",
          borderRadius: "9999px",
          backgroundColor: "rgba(13, 16, 23, 0.84)",
          backdropFilter: "blur(26px) saturate(190%)",
          WebkitBackdropFilter: "blur(26px) saturate(190%)",
          border: "1px solid rgba(255, 255, 255, 0.13)",
          boxShadow:
            "0 20px 48px -10px rgba(0, 0, 0, 0.82), inset 0 1px 1px 0 rgba(255, 255, 255, 0.2), inset 0 -1px 2px 0 rgba(0, 0, 0, 0.4)",
          overflow: "visible", // Allows the 78px spherical liquid glass orb to protrude freely
        }}
      >
        {NAV_ITEMS.map((item) => {
          const isActive = activeTab === item.id;
          const isHovered = hoveredTab === item.id;
          const Icon = item.icon;

          return (
            <motion.a
              key={item.id}
              href={item.href}
              onClick={(e) => handleClick(e, item)}
              onMouseEnter={() => setHoveredTab(item.id)}
              onMouseLeave={() => setHoveredTab(null)}
              layout
              transition={{
                type: "spring",
                stiffness: 380,
                damping: 30,
              }}
              className="liquid-nav-item"
              style={{
                position: "relative",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                width: isActive ? "78px" : "46px",
                height: "52px",
                textDecoration: "none",
                cursor: "pointer",
                userSelect: "none",
              }}
            >
              {/* Inactive Hover Floating Glass Micro-Tooltip */}
              <AnimatePresence>
                {isHovered && !isActive && (
                  <motion.div
                    initial={{ opacity: 0, y: 4, scale: 0.9 }}
                    animate={{ opacity: 1, y: -34, scale: 1 }}
                    exit={{ opacity: 0, y: 2, scale: 0.9 }}
                    transition={{ duration: 0.15 }}
                    style={{
                      position: "absolute",
                      top: 0,
                      padding: "4px 8px",
                      borderRadius: "6px",
                      backgroundColor: "rgba(10, 13, 18, 0.92)",
                      backdropFilter: "blur(14px)",
                      WebkitBackdropFilter: "blur(14px)",
                      border: "1px solid rgba(255, 255, 255, 0.18)",
                      color: "#ffffff",
                      fontSize: "11px",
                      fontWeight: 600,
                      whiteSpace: "nowrap",
                      pointerEvents: "none",
                      boxShadow: "0 8px 20px rgba(0, 0, 0, 0.6)",
                      zIndex: 30,
                    }}
                  >
                    {item.label}
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Active Protruding Spherical Liquid Glass Orb (Pushkar Verma iOS-26 Concept) */}
              {isActive && (
                <motion.div
                  layoutId="liquid-glass-active-orb"
                  transition={{
                    type: "spring",
                    stiffness: 350,
                    damping: 26,
                    mass: 0.75,
                  }}
                  style={{
                    position: "absolute",
                    top: "-13px",
                    left: 0,
                    width: "78px",
                    height: "78px",
                    borderRadius: "50%",
                    background:
                      "radial-gradient(circle at 50% 26%, #2b3442 0%, #151a24 52%, #090c11 100%)",
                    border: "1.5px solid rgba(255, 255, 255, 0.28)",
                    boxShadow: [
                      "0 18px 38px -6px rgba(0, 0, 0, 0.88)",
                      "0 6px 16px rgba(0, 0, 0, 0.55)",
                      "0 0 22px rgba(56, 189, 248, 0.25)",
                      "inset 0 1.5px 2.5px rgba(255, 255, 255, 0.92)",
                      "inset 0 -1.5px 2.5px rgba(255, 255, 255, 0.45)",
                      "inset 0 0 16px rgba(56, 189, 248, 0.18)",
                    ].join(", "),
                    zIndex: 10,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    pointerEvents: "none",
                  }}
                >
                  {/* Top Specular Meniscus Highlight Arc (Curved liquid lens glint with cyan refraction) */}
                  <div
                    style={{
                      position: "absolute",
                      top: "2px",
                      left: "14%",
                      right: "14%",
                      height: "24px",
                      borderRadius: "9999px",
                      background:
                        "radial-gradient(ellipse at 50% 0%, rgba(255, 255, 255, 0.98) 0%, rgba(56, 189, 248, 0.75) 45%, rgba(56, 189, 248, 0) 80%)",
                      filter: "blur(0.4px)",
                      pointerEvents: "none",
                    }}
                  />

                  {/* Bottom Specular Meniscus Highlight Arc (Curved caustics rim reflection) */}
                  <div
                    style={{
                      position: "absolute",
                      bottom: "2px",
                      left: "16%",
                      right: "16%",
                      height: "22px",
                      borderRadius: "9999px",
                      background:
                        "radial-gradient(ellipse at 50% 100%, rgba(255, 255, 255, 0.88) 0%, rgba(56, 189, 248, 0.7) 45%, rgba(56, 189, 248, 0) 80%)",
                      filter: "blur(0.4px)",
                      pointerEvents: "none",
                    }}
                  />

                  {/* Ambient Liquid Halo */}
                  <div
                    style={{
                      position: "absolute",
                      inset: "-4px",
                      borderRadius: "50%",
                      background:
                        "radial-gradient(circle, rgba(56, 189, 248, 0.35) 0%, rgba(56, 189, 248, 0) 70%)",
                      filter: "blur(8px)",
                      zIndex: -1,
                      pointerEvents: "none",
                    }}
                  />

                  {/* Active Content: White Prominent Icon + Crisp Label */}
                  <div
                    style={{
                      position: "relative",
                      zIndex: 15,
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      marginTop: "-1px",
                    }}
                  >
                    <Icon
                      size={21}
                      strokeWidth={2.3}
                      color="#ffffff"
                      style={{
                        filter: "drop-shadow(0 2px 5px rgba(0, 0, 0, 0.6))",
                      }}
                    />
                    <span
                      style={{
                        fontSize: "10.5px",
                        fontWeight: 600,
                        letterSpacing: "0.03em",
                        color: "#ffffff",
                        marginTop: "3px",
                        lineHeight: 1,
                        whiteSpace: "nowrap",
                        filter: "drop-shadow(0 1px 3px rgba(0, 0, 0, 0.7))",
                      }}
                    >
                      {item.label}
                    </span>
                  </div>
                </motion.div>
              )}

              {/* Inactive Icon (Only rendered when tab is not active) */}
              {!isActive && (
                <div
                  style={{
                    position: "relative",
                    zIndex: 1,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    transition: "all 0.2s ease",
                    transform: isHovered ? "scale(1.15)" : "scale(1)",
                    color: isHovered ? "#ffffff" : "rgba(203, 213, 225, 0.55)",
                  }}
                >
                  <Icon size={19} strokeWidth={isHovered ? 2.1 : 1.8} />
                </div>
              )}
            </motion.a>
          );
        })}
      </nav>
    </div>
  );
}

export default LiquidGlassNavbar;

