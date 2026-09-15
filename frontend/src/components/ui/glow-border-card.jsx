'use client';
import React from 'react';
import { cn } from '@/lib/utils';

// High-intensity multi-color neon spectrum matching Vengeance UI documentation
const DEFAULT_GLOW_COLORS = [
  '#00ff87',
  '#60efff',
  '#3b82f6',
  '#8b5cf6',
  '#ec4899',
  '#f43f5e',
  '#f59e0b',
  '#00ff87'
];

export const GlowBorderCard = React.forwardRef(function GlowBorderCard(
  {
    children,
    className,
    glowColors = DEFAULT_GLOW_COLORS,
    duration = 4,
    colorPreset,
    borderWidth = 3,
    borderRadius = 20,
    style,
    ...props
  },
  ref
) {
  const activeGlowColors = colorPreset === 'ocean'
    ? ['#006699', '#1177aa', '#2288bb', '#3399cc', '#44aadd', '#55bbee', '#66ccff', '#006699']
    : (glowColors && glowColors.length > 0 ? glowColors : DEFAULT_GLOW_COLORS);

  const gradientStops = activeGlowColors.join(', ');
  const innerRadius = Math.max(0, borderRadius - borderWidth);

  // Common styles for the rotating conic gradient square
  const spinnerStyle = {
    position: 'absolute',
    top: '50%',
    left: '50%',
    width: 'max(240vw, 2800px)',
    height: 'max(240vw, 2800px)',
    transform: 'translate(-50%, -50%)',
    transformOrigin: '50% 50%',
    background: `conic-gradient(from 0deg at 50% 50%, ${gradientStops})`,
    animation: `glow-card-spin ${duration}s linear infinite`,
    willChange: 'transform',
    pointerEvents: 'none',
  };

  return (
    <div
      ref={ref}
      className={cn(
        "relative group isolate",
        className
      )}
      style={{
        position: 'relative',
        borderRadius: `${borderRadius}px`,
        padding: `${borderWidth}px`,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'visible',
        ...style,
      }}
      {...props}
    >
      {/* 1. Outer Deep Atmospheric Bloom (Radiates 35px+ into surrounding dark canvas) */}
      <div
        className="glow-ambient-bloom"
        style={{
          position: 'absolute',
          inset: '-12px',
          borderRadius: `${borderRadius + 10}px`,
          filter: 'blur(38px)',
          opacity: 0.85,
          zIndex: 0,
          pointerEvents: 'none',
          overflow: 'hidden',
          willChange: 'transform',
        }}
      >
        <div style={spinnerStyle} />
      </div>

      {/* 2. Tight High-Intensity Neon Halo (Hugs the card boundary for extra pop) */}
      <div
        className="glow-halo-bloom"
        style={{
          position: 'absolute',
          inset: '-2px',
          borderRadius: `${borderRadius + 2}px`,
          filter: 'blur(10px)',
          opacity: 0.95,
          zIndex: 1,
          pointerEvents: 'none',
          overflow: 'hidden',
          willChange: 'transform',
        }}
      >
        <div style={spinnerStyle} />
      </div>

      {/* 3. Crisp Rotating Conic Border Track (Exact 3px concentric border) */}
      <div
        className="glow-crisp-border"
        style={{
          position: 'absolute',
          inset: 0,
          borderRadius: `${borderRadius}px`,
          overflow: 'hidden',
          zIndex: 2,
          pointerEvents: 'none',
          willChange: 'transform',
        }}
      >
        <div style={spinnerStyle} />
      </div>

      {/* 4. Inner Card Surface (Offset by padding, concentric corners, deep black contrast) */}
      <div
        className="glow-inner-card"
        style={{
          position: 'relative',
          zIndex: 10,
          flex: 1,
          width: '100%',
          backgroundColor: '#060b10',
          borderRadius: `${innerRadius}px`,
          padding: '24px',
          color: '#ffffff',
          overflow: 'hidden',
          boxShadow: '0 20px 50px rgba(0, 0, 0, 0.75)',
        }}
      >
        {children}
      </div>

      <style>{`
        @keyframes glow-card-spin {
          0% {
            transform: translate(-50%, -50%) rotate(0deg);
          }
          100% {
            transform: translate(-50%, -50%) rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
});

GlowBorderCard.displayName = 'GlowBorderCard';

export default GlowBorderCard;

