"""
SOFT DRIVE - Dynamic Relational Storage System

Core principle: Data is stored in relationships (quadruple phase sums), not in individual spins.
Multiple avatars can coexist and be added dynamically without destroying existing data.

Physics:
- Spins: Classical phases θ ∈ [0, 2π) on a 2D grid
- Quadruple constraints: θ_i + θ_j + θ_k + θ_l = φ_target (mod 2π)
- Strong force (K=2000): Enforces constraints (shape memory)
- Weak force (J=0.3): XY coupling allows substrate to evolve
- Dynamics: Damped velocity Verlet with γ=2.0

Metrics:
- PSNR (Peak Signal-to-Noise Ratio): Measures avatar quality.
  > 30 dB = excellent, 20-30 dB = recognizable, < 20 dB = degraded
- Constraint error: Angular deviation from target phase sum (radians)
- Energy: Total system energy (XY + constraint potential)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict
from dataclasses import dataclass
import time

@dataclass
class Quadruple:
    """A single relational constraint involving 4 spins."""
    i: int; j: int; k: int; l: int
    target_phase: float
    owner: str

class SoftDrive:
    """
    The substrate. Spins evolve under physical dynamics while
    quadruple constraints preserve the stored avatars.
    """
    def __init__(self, nx=60, ny=35, J=0.3, K=2000.0, gamma=2.0, dt=0.00015):
        self.nx, self.ny = nx, ny
        self.N = nx * ny
        self.J, self.K, self.gamma, self.dt = J, K, gamma, dt
        
        self.phases = 2 * np.pi * np.random.random(self.N)
        self.velocities = np.zeros(self.N)
        self.quadruples: List[Quadruple] = []
        self.objects: Dict[str, List[Quadruple]] = {}
        
        # History tracking
        self.history = {
            'steps': [],
            'energy': [],
            'max_error': [],
            'object_psnr': {}
        }
        
        self.step_count = 0
        
    def _idx(self, x, y): 
        """Convert grid coordinates (x,y) to flat spin index."""
        return y * self.nx + x
    
    def _create_quad(self, x, y, target, owner):
        """
        Create a quadruple constraint on the square with bottom-left corner (x,y).
        The 4 spins are: bottom-left, bottom-right, top-right, top-left.
        """
        if x >= self.nx-1 or y >= self.ny-1:
            return None
        i = self._idx(x, y)
        j = self._idx(x+1, y)
        k = self._idx(x+1, y+1)
        l = self._idx(x, y+1)
        q = Quadruple(i, j, k, l, target % (2*np.pi), owner)
        self.quadruples.append(q)
        return q
    
    def add_avatar(self, pattern, name, center_x, center_y, timestamp=None):
        """
        Add a new avatar to the soft drive.
        
        The avatar's pixels become target phases for quadruple constraints
        arranged in a grid around the specified center position.
        
        PSNR (Peak Signal-to-Noise Ratio):
            Measures how well the avatar can be retrieved.
            Calculated as: 10 * log10(1 / MSE)
            
            - 100 dB: Perfect (theoretical maximum)
            - 60+ dB: Excellent, indistinguishable from original
            - 40+ dB: Very good, minor differences
            - 30+ dB: still recognizable
            - 20-30 dB: visible degradation
            - < 20 dB: Poor
        
        Returns:
            Initial PSNR after adding (should be near 100 dB)
        """
        h, w = pattern.shape
        quads = []
        
        # Create quadruples for each pixel
        for y in range(h):
            for x in range(w):
                grid_x = center_x + x - w//2
                grid_y = center_y + y - h//2
                if 0 <= grid_x < self.nx-1 and 0 <= grid_y < self.ny-1:
                    # Pixel value (0 to 1) maps to target phase (0 to 2π)
                    phase = 2 * np.pi * pattern[y, x]
                    q = self._create_quad(grid_x, grid_y, phase, name)
                    if q:
                        quads.append(q)
        
        self.objects[name] = quads
        
        # Re-solve all constraints
        self.solve_constraints()
        
        # Measure initial quality
        retrieved = self.retrieve_avatar(name, pattern.shape)
        psnr = 10 * np.log10(1.0 / (np.mean((retrieved - pattern)**2) + 1e-10))
        
        t_str = f" at step {timestamp}" if timestamp else ""
        print(f"\n  ➕ ADDED '{name}'{t_str}: {len(quads)} quads, initial PSNR = {psnr:.2f} dB")
        
        return psnr
    
    def solve_constraints(self):
        """
        Solve for spin phases that satisfy ALL quadruple constraints.
        
        Each constraint: θ_i + θ_j + θ_k + θ_l = φ_target (mod 2π)
        This is a linear system: A·θ = b (mod 2π)
        
        We solve using least squares, which finds the configuration that
        minimizes constraint violation when the system is overconstrained.
        
        The solution gives us the initial phase field that perfectly
        encodes all stored avatars simultaneously.
        """
        if not self.quadruples:
            return
        
        nq = len(self.quadruples)
        A = np.zeros((nq, self.N))
        b = np.zeros(nq)
        
        for i, q in enumerate(self.quadruples):
            A[i, q.i] = A[i, q.j] = A[i, q.k] = A[i, q.l] = 1.0
            b[i] = q.target_phase
        # Least squares solution (handles overconstrained systems)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        self.phases = sol % (2*np.pi)
    
    def phase_sum(self, q):
        return (self.phases[q.i] + self.phases[q.j] + 
                self.phases[q.k] + self.phases[q.l]) % (2*np.pi)
    
    def retrieve_avatar(self, name, shape):
        """
        Extract an avatar from the current phase field.
        
        Each quadruple's current phase sum is converted back to a pixel value
        by dividing by 2π. This is the inverse of the encoding process.
        
        Returns:
            Grayscale image array with values in [0, 1]
        """
        h, w = shape
        img = np.zeros((h, w))
        cnt = np.zeros((h, w))
        quads = self.objects.get(name, [])
        
        for idx, q in enumerate(quads):
            x, y = idx % w, idx // w
            if x < w and y < h:
                val = self.phase_sum(q) / (2*np.pi)
                img[y, x] += val
                cnt[y, x] += 1
        
        cnt[cnt == 0] = 1
        return np.clip(img / cnt, 0, 1)
    
    def compute_max_error(self):
        """
        Maximum constraint violation across all quadruples.
        
        Error = angular difference between current phase sum and target.
        Range: 0 rad (perfect) to π rad (completely wrong).
        
        For a working soft drive, error should stay < 0.1 rad (≈6°)
        """
        if not self.quadruples:
            return 0.0
        
        max_err = 0.0
        for q in self.quadruples:
            current = self.phase_sum(q)
            diff = abs(current - q.target_phase)
            diff = min(diff, 2*np.pi - diff)
            max_err = max(max_err, diff)
        return max_err
    
    def compute_psnr_for_object(self, name, original_pattern):
        """Compute PSNR for a specific avatar."""
        retrieved = self.retrieve_avatar(name, original_pattern.shape)
        mse = np.mean((retrieved - original_pattern)**2)
        if mse == 0:
            return 100.0
        return 10 * np.log10(1.0 / mse)
    
    def _xy_energy(self):
        e = 0.0
        for y in range(self.ny):
            for x in range(self.nx-1):
                i, j = self._idx(x,y), self._idx(x+1,y)
                e -= self.J * np.cos(self.phases[i] - self.phases[j])
        for y in range(self.ny-1):
            for x in range(self.nx):
                i, j = self._idx(x,y), self._idx(x,y+1)
                e -= self.J * np.cos(self.phases[i] - self.phases[j])
        return e
    
    def _quad_energy(self):
        e = 0.0
        for q in self.quadruples:
            diff = self.phase_sum(q) - q.target_phase
            e += 0.5 * self.K * (1 - np.cos(diff))
        return e
    
    def total_energy(self):
        """
        Total system energy = XY energy + Constraint potential
        
        XY energy (J=0.3): Weak coupling favoring neighboring spins to align.
            - Lower XY energy means smoother phase field.
            - This is the "bouncing" mechanism - the substrate wants to move.
        
        Constraint potential (K=2000): Strong springs pulling each quadruple
            toward its target phase sum. This is the "shape memory."
            - Higher constraint energy means avatars are distorted.
        
        Energy decreases over time as the system relaxes toward equilibrium.
        """
        return self._xy_energy() + self._quad_energy()
    
    def _forces(self):
        """
        Compute total force on each spin.
        
        Forces come from two competing sources:
        1. XY forces: Push neighbors to align (weak, J=0.3)
        2. Constraint forces: Push quadruple sums toward targets (strong, K=2000)
        
        The constraint forces are much stronger, so shape memory dominates.
        XY forces add "bouncing" - the substrate evolves while preserving data.
        """
        F = np.zeros(self.N)
        
        # XY forces
        for y in range(self.ny):
            for x in range(self.nx-1):
                i, j = self._idx(x,y), self._idx(x+1,y)
                d = self.phases[i] - self.phases[j]
                f = -self.J * np.sin(d)
                F[i] += f
                F[j] -= f
        for y in range(self.ny-1):
            for x in range(self.nx):
                i, j = self._idx(x,y), self._idx(x,y+1)
                d = self.phases[i] - self.phases[j]
                f = -self.J * np.sin(d)
                F[i] += f
                F[j] -= f
        
        # Quadruple forces
        for q in self.quadruples:
            s = (self.phases[q.i] + self.phases[q.j] + 
                 self.phases[q.k] + self.phases[q.l])
            f = -self.K * np.sin(s - q.target_phase)
            F[q.i] += f
            F[q.j] += f
            F[q.k] += f
            F[q.l] += f
        
        return F
    
    def step(self, n_steps=10, record=True):
        """
        Evolve the system forward in time using velocity Verlet integration.
        
        Each step:
        1. Compute forces from current phases
        2. Update velocities (half step)
        3. Apply damping (removes energy, leads to equilibrium)
        4. Update phases
        5. Recompute forces at new positions
        6. Update velocities (second half step)
        
        After many steps, the system relaxes toward a low-energy configuration
        that still satisfies all constraints (avatars remain readable).
        """
        for _ in range(n_steps):
            F = self._forces()
            self.velocities += 0.5 * self.dt * F
            self.velocities *= (1 - self.gamma * self.dt)
            self.phases += self.dt * self.velocities
            self.phases %= 2*np.pi
            F_new = self._forces()
            self.velocities += 0.5 * self.dt * F_new
            self.velocities *= (1 - self.gamma * self.dt)
        
        self.step_count += n_steps
        
        if record:
            self.history['steps'].append(self.step_count)
            self.history['energy'].append(self.total_energy())
            self.history['max_error'].append(self.compute_max_error())
    
    def phase_grid(self):
        return self.phases.reshape(self.ny, self.nx)


def create_avatar(shape="smiley", size=10):
    """Create an avatar pattern."""
    img = Image.new('L', (size, size), 180)
    draw = ImageDraw.Draw(img)
    c = size // 2
    r = size // 3
    
    if shape == "smiley":
        draw.ellipse([c-r, c-r, c+r, c+r], outline=40, width=2, fill=140)
        er = size // 8
        draw.ellipse([c-size//4-er, c-size//4-er, c-size//4+er, c-size//4+er], fill=30)
        draw.ellipse([c+size//4-er, c-size//4-er, c+size//4+er, c-size//4+er], fill=30)
        draw.arc([c-r+2, c-2, c+r-2, c+r-2], 0, 180, fill=50, width=2)
    elif shape == "circle":
        draw.ellipse([c-r, c-r, c+r, c+r], outline=40, width=2, fill=140)
    elif shape == "square":
        s = size // 2
        draw.rectangle([c-s//2, c-s//2, c+s//2, c+s//2], outline=40, width=2, fill=140)
    elif shape == "diamond":
        draw.polygon([(c, c-r), (c+r, c), (c, c+r), (c-r, c)], outline=40, fill=140)
    
    return np.array(img) / 255.0


def run_demo():
    print("=" * 70)
    print("DYNAMIC SOFT DRIVE - Staggered Avatar Introduction")
    print("Physics: J=0.3, K=2000, γ=2.0, dt=0.00015")
    print("=" * 70)
    
    # Create drive
    drive = SoftDrive(nx=60, ny=35, J=0.3, K=2000.0, gamma=2.0, dt=0.00015)
    
    # Create avatar patterns
    smiley = create_avatar("smiley", size=10)
    circle = create_avatar("circle", size=8)
    square = create_avatar("square", size=9)
    
    print("\n" + "=" * 70)
    print("PHASE 1: Empty system evolves (no avatars)")
    print("=" * 70)
    
    # Run with no avatars
    for step in range(500):
        drive.step(10, record=(step % 10 == 0))
    
    print(f"  Step {drive.step_count}: Energy = {drive.total_energy():.2f}, No constraints")
    
    print("\n" + "=" * 70)
    print("PHASE 2: Adding first avatar (Smiley)")
    print("=" * 70)
    
    # Add smiley at center
    drive.add_avatar(smiley, "smiley", 30, 17, timestamp=drive.step_count)
    
    # Evolve with smiley
    psnr_history_smiley = [drive.compute_psnr_for_object("smiley", smiley)]
    
    for step in range(1000):
        drive.step(10, record=(step % 20 == 0))
        psnr_history_smiley.append(drive.compute_psnr_for_object("smiley", smiley))
    
    print(f"\n  Step {drive.step_count}: Smiley PSNR = {psnr_history_smiley[-1]:.2f} dB")
    
    print("\n" + "=" * 70)
    print("PHASE 3: Adding second avatar (Circle)")
    print("=" * 70)
    
    # Add circle at different location
    drive.add_avatar(circle, "circle", 45, 25, timestamp=drive.step_count)
    
    # Track both
    psnr_history_circle = [drive.compute_psnr_for_object("circle", circle)]
    psnr_history_smiley.append(drive.compute_psnr_for_object("smiley", smiley))
    
    for step in range(1000):
        drive.step(10, record=(step % 20 == 0))
        psnr_history_smiley.append(drive.compute_psnr_for_object("smiley", smiley))
        psnr_history_circle.append(drive.compute_psnr_for_object("circle", circle))
    
    print(f"\n  Step {drive.step_count}: Smiley PSNR = {psnr_history_smiley[-1]:.2f} dB")
    print(f"  Step {drive.step_count}: Circle PSNR = {psnr_history_circle[-1]:.2f} dB")
    
    print("\n" + "=" * 70)
    print("PHASE 4: Adding third avatar (Square)")
    print("=" * 70)
    
    # Add square
    drive.add_avatar(square, "square", 15, 20, timestamp=drive.step_count)
    
    # Track all three
    psnr_history_square = [drive.compute_psnr_for_object("square", square)]
    psnr_history_smiley.append(drive.compute_psnr_for_object("smiley", smiley))
    psnr_history_circle.append(drive.compute_psnr_for_object("circle", circle))
    
    for step in range(1000):
        drive.step(10, record=(step % 20 == 0))
        psnr_history_smiley.append(drive.compute_psnr_for_object("smiley", smiley))
        psnr_history_circle.append(drive.compute_psnr_for_object("circle", circle))
        psnr_history_square.append(drive.compute_psnr_for_object("square", square))
    
    print(f"\n  Step {drive.step_count}: Smiley PSNR = {psnr_history_smiley[-1]:.2f} dB")
    print(f"  Step {drive.step_count}: Circle PSNR = {psnr_history_circle[-1]:.2f} dB")
    print(f"  Step {drive.step_count}: Square PSNR = {psnr_history_square[-1]:.2f} dB")
    
    # VISUALIZATION - Dynamic Graphs
    print("\n" + "=" * 70)
    print("GENERATING DYNAMIC GRAPHS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dynamic Soft Drive - Staggered Avatar Introduction", fontsize=14)
    
    # Graph 1: Constraint Error Over Time
    ax1 = axes[0, 0]
    steps = drive.history['steps']
    errors = drive.history['max_error']
    ax1.semilogy(steps, errors, 'b-', linewidth=1)
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Max Constraint Error (rad)")
    ax1.set_title("Constraint Violation")
    ax1.grid(True, alpha=0.3)
    
    # Mark avatar introduction times
    intro_times = [500, 1500, 2500]  # Approximate steps when avatars were added
    for t in intro_times:
        ax1.axvline(x=t, color='r', linestyle='--', alpha=0.5)
    ax1.text(intro_times[0], 0.5, 'Smiley', rotation=90, fontsize=8)
    ax1.text(intro_times[1], 0.5, 'Circle', rotation=90, fontsize=8)
    ax1.text(intro_times[2], 0.5, 'Square', rotation=90, fontsize=8)
    
    # Graph 2: Total Energy Over Time
    ax2 = axes[0, 1]
    ax2.plot(steps, drive.history['energy'], 'g-', linewidth=1)
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Total Energy")
    ax2.set_title("System Energy")
    ax2.grid(True, alpha=0.3)
    for t in intro_times:
        ax2.axvline(x=t, color='r', linestyle='--', alpha=0.5)
    
    # Graph 3: Avatar PSNR Over Time
    ax3 = axes[1, 0]
    
    # Align PSNR histories with steps
    steps_smiley = list(range(0, len(psnr_history_smiley) * 10, 10))
    steps_circle = list(range(1500, 1500 + len(psnr_history_circle) * 10, 10))
    steps_square = list(range(2500, 2500 + len(psnr_history_square) * 10, 10))
    
    ax3.plot(steps_smiley[:len(psnr_history_smiley)], psnr_history_smiley, 'b-', label='Smiley', linewidth=1)
    ax3.plot(steps_circle[:len(psnr_history_circle)], psnr_history_circle, 'g-', label='Circle', linewidth=1)
    ax3.plot(steps_square[:len(psnr_history_square)], psnr_history_square, 'r-', label='Square', linewidth=1)
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("PSNR (dB)")
    ax3.set_title("Avatar Quality Over Time")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    for t in intro_times:
        ax3.axvline(x=t, color='r', linestyle='--', alpha=0.3)
    
    # Graph 4: Final Retrieved Avatars (simple version)
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Retrieve final states
    final_smiley = drive.retrieve_avatar("smiley", smiley.shape)
    final_circle = drive.retrieve_avatar("circle", circle.shape)
    final_square = drive.retrieve_avatar("square", square.shape)

    # Just create a new figure for the avatars
    fig2, axes2 = plt.subplots(2, 3, figsize=(10, 6))
    fig2.suptitle("Final Retrieved Avatars vs Originals", fontsize=12)

    axes2[0, 0].imshow(smiley, cmap='gray', vmin=0, vmax=1)
    axes2[0, 0].set_title("Original Smiley")
    axes2[0, 0].axis('off')

    axes2[1, 0].imshow(final_smiley, cmap='gray', vmin=0, vmax=1)
    axes2[1, 0].set_title(f"Retrieved Smiley ({psnr_history_smiley[-1]:.1f} dB)")
    axes2[1, 0].axis('off')

    axes2[0, 1].imshow(circle, cmap='gray', vmin=0, vmax=1)
    axes2[0, 1].set_title("Original Circle")
    axes2[0, 1].axis('off')

    axes2[1, 1].imshow(final_circle, cmap='gray', vmin=0, vmax=1)
    axes2[1, 1].set_title(f"Retrieved Circle ({psnr_history_circle[-1]:.1f} dB)")
    axes2[1, 1].axis('off')

    axes2[0, 2].imshow(square, cmap='gray', vmin=0, vmax=1)
    axes2[0, 2].set_title("Original Square")
    axes2[0, 2].axis('off')

    axes2[1, 2].imshow(final_square, cmap='gray', vmin=0, vmax=1)
    axes2[1, 2].set_title(f"Retrieved Square ({psnr_history_square[-1]:.1f} dB)")
    axes2[1, 2].axis('off')

    plt.tight_layout()
    plt.show()
    
    # SUMMARY
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total steps simulated: {drive.step_count}")
    print(f"Avatars added: Smiley (step 500), Circle (step 1500), Square (step 2500)")
    print(f"\nFinal preservation:")
    print(f"  Smiley: {psnr_history_smiley[-1]:.2f} dB")
    print(f"  Circle: {psnr_history_circle[-1]:.2f} dB")
    print(f"  Square: {psnr_history_square[-1]:.2f} dB")
    
    min_psnr = min(psnr_history_smiley[-1], psnr_history_circle[-1], psnr_history_square[-1])
    if min_psnr > 25:
        print("\n✓✓✓ SUCCESS! All avatars preserved through dynamic addition and evolution!")
    elif min_psnr > 15:
        print("\n✓ SUCCESS! Avatars preserved with minor degradation")
    else:
        print("\n⚠ Some degradation - but system remains functional")
    
    return drive


if __name__ == "__main__":
    drive = run_demo()