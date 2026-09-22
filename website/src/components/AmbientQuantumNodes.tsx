import React, { useEffect, useRef } from 'react';

export const AmbientQuantumNodes: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return;
      const x = (e.clientX / window.innerWidth - 0.5) * 20;
      const y = (e.clientY / window.innerHeight - 0.5) * 20;
      containerRef.current.style.transform = `translate3d(${x}px, ${y}px, 0)`;
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="ambient-quantum-backdrop" aria-hidden="true" ref={containerRef}>
      <div className="quantum-orb orb-1"></div>
      <div className="quantum-orb orb-2"></div>
      <div className="quantum-orb orb-3"></div>
      <div className="quantum-laser-grid"></div>
    </div>
  );
};
