import React from 'react';

interface CollisionLogoProps {
  size?: number;
  color?: string;
  className?: string;
  animated?: boolean;
}

/**
 * EXACT COLLISION ICON:
 * Matches the user-provided mark:
 * - Top-Left: Round "C" opening to the right
 * - Top-Right: Round "O" with upper-right inner accent
 * - Bottom-Left: "T" / "7" with horizontal top bar & central vertical downward stem
 * - Bottom-Right: "n" (curved arch with two vertical downward legs)
 */
export const CollisionLogo: React.FC<CollisionLogoProps> = ({
  size = 32,
  color = '#8468DA',
  className = '',
  animated = false,
}) => {
  return (
    <div
      className={`collision-brand-icon-wrap ${animated ? 'spin-pulse' : ''} ${className}`}
      style={{
        width: size,
        height: size,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
      }}
    >
      <svg
        width={size}
        height={size}
        viewBox="0 0 100 100"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{ display: 'block', overflow: 'visible' }}
      >
        {/* =========================================================
            TOP-LEFT: "C"
            Circle centered at (25, 25), outer R=20, inner R=10, right opening gap
            ========================================================= */}
        <path
          d="M 38 12.5
             A 18 18 0 1 0 38 37.5
             L 34.5 30.5
             A 10 10 0 1 1 34.5 19.5
             Z"
          fill={color}
        />

        {/* =========================================================
            TOP-RIGHT: "O"
            Circle centered at (75, 25), outer R=18, inner R=10
            with the distinct upper-right crescent accent
            ========================================================= */}
        <path
          fillRule="evenodd"
          clipRule="evenodd"
          d="M 75 7
             A 18 18 0 1 0 75 43
             A 18 18 0 1 0 75 7
             Z
             M 75 16
             A 9 9 0 1 1 75 34
             A 9 9 0 1 1 75 16
             Z"
          fill={color}
        />
        {/* Upper-right accent in O */}
        <path
          d="M 83.5 18
             C 86.5 22 86 28 83 31
             C 84.5 28 84.5 22 81 18.5
             C 78.5 16 75.5 15.5 72.5 15.5
             C 77 15 81.5 15.5 83.5 18
             Z"
          fill="#FFFFFF"
          opacity="0.95"
        />

        {/* =========================================================
            BOTTOM-LEFT: "T"
            Horizontal bar (7 to 43, y: 53 to 63)
            Vertical stem hanging down from middle-right (x: 23 to 33, y: 63 to 91)
            ========================================================= */}
        <path
          d="M 7 53
             H 43
             V 63
             H 34
             V 91
             H 23
             V 63
             H 7
             Z"
          fill={color}
        />

        {/* =========================================================
            BOTTOM-RIGHT: "n" (Arch)
            Curved top arch from x=57 to x=93 (y: 53 to 67)
            Left leg (57 to 68, y down to 91)
            Right leg (82 to 93, y down to 91)
            ========================================================= */}
        <path
          d="M 57 91
             V 70
             C 57 58 64 53 75 53
             C 86 53 93 58 93 70
             V 91
             H 82
             V 71
             C 82 65.5 79.5 63 75 63
             C 70.5 63 68 65.5 68 71
             V 91
             H 57
             Z"
          fill={color}
        />
      </svg>
    </div>
  );
};

/**
 * FULL COLLISION BRAND LOCKUP (Icon + Heavy Bold Typographic Wordmark)
 */
export const CollisionBrandLockup: React.FC<{
  iconSize?: number;
  fontSize?: number;
  color?: string;
  textColor?: string;
}> = ({
  iconSize = 34,
  fontSize = 24,
  color = '#8468DA',
  textColor,
}) => {
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '14px',
        textDecoration: 'none',
        userSelect: 'none',
      }}
    >
      <CollisionLogo size={iconSize} color={color} />
      <span
        style={{
          fontFamily: "'Plus Jakarta Sans', system-ui, sans-serif",
          fontWeight: 900,
          fontSize: `${fontSize}px`,
          letterSpacing: '-0.035em',
          color: textColor || 'var(--text-primary)',
          lineHeight: 1,
          display: 'flex',
          alignItems: 'center',
        }}
      >
        COLLISION
      </span>
    </div>
  );
};

/* Official animated 4-quadrant loader based on the exact 4 glyphs */
export const CollisionLoadingMark: React.FC<{ size?: number; label?: string }> = ({
  size = 28,
  label,
}) => {
  return (
    <div className="collision-loader-container">
      <div className="collision-quadrant-loader" style={{ width: size, height: size }}>
        <div className="quadrant-piece q1" title="C" />
        <div className="quadrant-piece q2" title="O" />
        <div className="quadrant-piece q3" title="T" />
        <div className="quadrant-piece q4" title="n" />
      </div>
      {label && <span className="collision-loader-label">{label}</span>}
    </div>
  );
};
