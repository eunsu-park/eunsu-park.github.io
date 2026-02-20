"""
Panorama Image Stitching

This script demonstrates panorama creation from multiple images using:
- Feature detection (SIFT, ORB)
- Feature matching (BFMatcher, FLANN)
- Homography estimation with RANSAC
- Image warping and blending

Requirements:
    pip install opencv-contrib-python numpy

Author: Claude
Date: 2026-02-15
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import sys


class PanoramaStitcher:
    """
    Complete pipeline for creating panoramas from multiple images.

    Supports various feature detectors and matchers, with configurable
    homography estimation and blending strategies.
    """

    def __init__(self,
                 detector: str = 'sift',
                 matcher: str = 'flann',
                 ratio_thresh: float = 0.75,
                 ransac_reproj_thresh: float = 5.0,
                 blend_mode: str = 'linear'):
        """
        Args:
            detector: Feature detector ('sift', 'orb')
            matcher: Feature matcher ('flann', 'bf')
            ratio_thresh: Ratio test threshold for Lowe's ratio test
            ransac_reproj_thresh: RANSAC reprojection error threshold
            blend_mode: Blending mode ('linear', 'multiband')
        """
        self.detector_type = detector.lower()
        self.matcher_type = matcher.lower()
        self.ratio_thresh = ratio_thresh
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.blend_mode = blend_mode

        # Initialize detector
        self.detector = self._create_detector()

        # Initialize matcher
        self.matcher = self._create_matcher()

    def _create_detector(self):
        """Create feature detector."""
        if self.detector_type == 'sift':
            try:
                return cv2.SIFT_create()
            except AttributeError:
                print("SIFT not available, falling back to ORB")
                self.detector_type = 'orb'
                return cv2.ORB_create(nfeatures=2000)
        elif self.detector_type == 'orb':
            return cv2.ORB_create(nfeatures=2000)
        else:
            raise ValueError(f"Unknown detector: {self.detector_type}")

    def _create_matcher(self):
        """Create feature matcher."""
        if self.matcher_type == 'flann':
            if self.detector_type == 'sift':
                # FLANN parameters for SIFT
                index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
                search_params = dict(checks=50)
            else:
                # FLANN parameters for ORB
                index_params = dict(
                    algorithm=6,  # FLANN_INDEX_LSH
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1
                )
                search_params = dict(checks=50)

            return cv2.FlannBasedMatcher(index_params, search_params)
        elif self.matcher_type == 'bf':
            # BFMatcher with appropriate norm
            norm_type = cv2.NORM_L2 if self.detector_type == 'sift' else cv2.NORM_HAMMING
            return cv2.BFMatcher(norm_type, crossCheck=False)
        else:
            raise ValueError(f"Unknown matcher: {self.matcher_type}")

    def detect_and_describe(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect keypoints and compute descriptors.

        Args:
            image: Input image (grayscale or color)

        Returns:
            Tuple of (keypoints, descriptors)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self,
                      desc1: np.ndarray,
                      desc2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match features between two images using ratio test.

        Args:
            desc1, desc2: Feature descriptors

        Returns:
            List of good matches
        """
        # Ensure descriptors are in correct format
        if self.detector_type == 'orb' and desc1.dtype != np.uint8:
            desc1 = desc1.astype(np.uint8)
            desc2 = desc2.astype(np.uint8)
        elif self.detector_type == 'sift' and desc1.dtype != np.float32:
            desc1 = desc1.astype(np.float32)
            desc2 = desc2.astype(np.float32)

        # Find k=2 best matches for ratio test
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_thresh * n.distance:
                    good_matches.append(m)

        return good_matches

    def estimate_homography(self,
                          kp1: List,
                          kp2: List,
                          matches: List[cv2.DMatch]) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Estimate homography matrix using RANSAC.

        Args:
            kp1, kp2: Keypoints from both images
            matches: Good matches

        Returns:
            Tuple of (homography matrix, mask of inliers)
        """
        if len(matches) < 4:
            print(f"Insufficient matches: {len(matches)} (need at least 4)")
            return None, None

        # Extract matched keypoint coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Compute homography with RANSAC
        H, mask = cv2.findHomography(
            pts1, pts2,
            cv2.RANSAC,
            self.ransac_reproj_thresh
        )

        if H is None:
            print("Homography estimation failed")
            return None, None

        # Check for degenerate homography
        if not self._is_valid_homography(H):
            print("Degenerate homography detected")
            return None, None

        return H, mask

    def _is_valid_homography(self, H: np.ndarray) -> bool:
        """Check if homography is valid (not degenerate)."""
        # Check determinant
        det = np.linalg.det(H)
        if abs(det) < 1e-6:
            return False

        # Check if transformation is too extreme
        # (prevents warping to infinity)
        h_norm = H / H[2, 2]  # Normalize
        if abs(h_norm[2, 0]) > 0.001 or abs(h_norm[2, 1]) > 0.001:
            # Perspective component too large
            return False

        return True

    def warp_images(self,
                   img1: np.ndarray,
                   img2: np.ndarray,
                   H: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Warp img1 to align with img2 using homography H.

        Args:
            img1: Source image to warp
            img2: Target/reference image
            H: Homography matrix

        Returns:
            Tuple of (warped panorama, offset of img2 in panorama)
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Get corners of img1
        corners1 = np.float32([
            [0, 0],
            [w1, 0],
            [w1, h1],
            [0, h1]
        ]).reshape(-1, 1, 2)

        # Get corners of img2
        corners2 = np.float32([
            [0, 0],
            [w2, 0],
            [w2, h2],
            [0, h2]
        ]).reshape(-1, 1, 2)

        # Transform corners of img1
        corners1_transformed = cv2.perspectiveTransform(corners1, H)

        # Combine all corners
        all_corners = np.concatenate((corners2, corners1_transformed), axis=0)

        # Find bounding box
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        # Translation to bring all points into positive coordinates
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])

        # Warp img1
        output_size = (x_max - x_min, y_max - y_min)
        img1_warped = cv2.warpPerspective(
            img1,
            translation @ H,
            output_size,
            flags=cv2.INTER_LINEAR
        )

        # Position where img2 will be placed
        img2_offset = (-x_min, -y_min)

        return img1_warped, img2_offset

    def blend_images(self,
                    img1_warped: np.ndarray,
                    img2: np.ndarray,
                    offset: Tuple[int, int]) -> np.ndarray:
        """
        Blend two images using specified blending mode.

        Args:
            img1_warped: Warped first image
            img2: Second image
            offset: Position to place img2 in the panorama

        Returns:
            Blended panorama
        """
        panorama = img1_warped.copy()
        x_offset, y_offset = offset
        h2, w2 = img2.shape[:2]

        if self.blend_mode == 'linear':
            # Linear blending in overlap region
            # Create mask for img2
            mask2 = np.zeros(panorama.shape[:2], dtype=np.float32)
            mask2[y_offset:y_offset+h2, x_offset:x_offset+w2] = 1.0

            # Create mask for img1 (existing content)
            mask1 = (img1_warped.sum(axis=2) > 0).astype(np.float32)

            # Find overlap region
            overlap = (mask1 * mask2) > 0

            if overlap.any():
                # Compute distance transforms for smooth blending
                dist1 = cv2.distanceTransform((mask1 > 0).astype(np.uint8),
                                             cv2.DIST_L2, 5)
                dist2 = cv2.distanceTransform((mask2 > 0).astype(np.uint8),
                                             cv2.DIST_L2, 5)

                # Normalize distances in overlap region
                total_dist = dist1 + dist2 + 1e-6
                alpha1 = dist1 / total_dist
                alpha2 = dist2 / total_dist

                # Blend
                for c in range(3):
                    panorama[overlap, c] = (
                        alpha1[overlap] * img1_warped[overlap, c] +
                        alpha2[overlap] * img2[y_offset:y_offset+h2,
                                               x_offset:x_offset+w2][overlap[y_offset:y_offset+h2,
                                                                            x_offset:x_offset+w2], c]
                    ).astype(np.uint8)

            # Place img2 in non-overlap region
            non_overlap = (mask2 > 0) & (mask1 == 0)
            panorama[non_overlap] = img2[non_overlap[y_offset:y_offset+h2,
                                                     x_offset:x_offset+w2]]

        else:  # Simple alpha blending
            panorama[y_offset:y_offset+h2, x_offset:x_offset+w2] = img2

        return panorama

    def stitch(self,
              images: List[np.ndarray],
              visualize: bool = False) -> Optional[np.ndarray]:
        """
        Stitch multiple images into a panorama.

        Args:
            images: List of images to stitch (left to right order)
            visualize: Whether to show intermediate results

        Returns:
            Stitched panorama or None if failed
        """
        if len(images) < 2:
            print("Need at least 2 images to stitch")
            return None

        # Start with the middle image as base
        result = images[len(images) // 2].copy()

        # Stitch images to the right
        for i in range(len(images) // 2 + 1, len(images)):
            print(f"\nStitching image {i+1}/{len(images)}...")
            result = self._stitch_pair(result, images[i], visualize)
            if result is None:
                print(f"Failed to stitch image {i+1}")
                return None

        # Stitch images to the left (reverse order)
        for i in range(len(images) // 2 - 1, -1, -1):
            print(f"\nStitching image {i+1}/{len(images)}...")
            result = self._stitch_pair(images[i], result, visualize)
            if result is None:
                print(f"Failed to stitch image {i+1}")
                return None

        return result

    def _stitch_pair(self,
                    img1: np.ndarray,
                    img2: np.ndarray,
                    visualize: bool = False) -> Optional[np.ndarray]:
        """Stitch a pair of images."""
        # Detect and describe
        kp1, desc1 = self.detect_and_describe(img1)
        kp2, desc2 = self.detect_and_describe(img2)

        print(f"  Features: {len(kp1)} (left), {len(kp2)} (right)")

        if desc1 is None or desc2 is None:
            print("  Feature detection failed")
            return None

        # Match features
        matches = self.match_features(desc1, desc2)
        print(f"  Good matches: {len(matches)}")

        if visualize:
            matches_img = cv2.drawMatches(
                img1, kp1, img2, kp2, matches[:50], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.imshow('Feature Matches', matches_img)
            cv2.waitKey(500)

        # Estimate homography
        H, mask = self.estimate_homography(kp1, kp2, matches)

        if H is None:
            return None

        inliers = np.sum(mask) if mask is not None else 0
        print(f"  Inliers: {inliers}/{len(matches)}")

        # Warp and blend
        img1_warped, offset = self.warp_images(img1, img2, H)

        if visualize:
            cv2.imshow('Warped Image', img1_warped)
            cv2.waitKey(500)

        result = self.blend_images(img1_warped, img2, offset)

        return result


def generate_synthetic_images(num_images: int = 3,
                             base_size: int = 400,
                             overlap: float = 0.3) -> List[np.ndarray]:
    """
    Generate synthetic images for testing panorama stitching.

    Creates images with a checkerboard pattern that are shifted/rotated
    versions of each other.

    Args:
        num_images: Number of images to generate
        base_size: Base image size
        overlap: Overlap ratio between consecutive images

    Returns:
        List of generated images
    """
    images = []

    # Create base pattern (checkerboard with gradients)
    full_width = int(base_size * (1 + (num_images - 1) * (1 - overlap)))
    canvas = np.zeros((base_size, full_width, 3), dtype=np.uint8)

    # Draw checkerboard
    square_size = 40
    for i in range(0, base_size, square_size):
        for j in range(0, full_width, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                canvas[i:i+square_size, j:j+square_size] = [100, 100, 100]
            else:
                canvas[i:i+square_size, j:j+square_size] = [200, 200, 200]

    # Add some features (circles, rectangles)
    for i in range(num_images * 3):
        x = np.random.randint(50, full_width - 50)
        y = np.random.randint(50, base_size - 50)
        color = tuple(np.random.randint(0, 255, 3).tolist())

        if np.random.random() > 0.5:
            cv2.circle(canvas, (x, y), 20, color, -1)
        else:
            cv2.rectangle(canvas, (x-15, y-15), (x+15, y+15), color, -1)

    # Add text
    for i in range(num_images):
        x = int(i * full_width / num_images) + 50
        cv2.putText(canvas, f'Region {i+1}', (x, base_size // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Extract overlapping regions
    step = int(base_size * (1 - overlap))
    for i in range(num_images):
        x_start = i * step
        x_end = x_start + base_size
        img = canvas[:, x_start:x_end].copy()

        # Add slight rotation and noise for realism
        if i > 0:
            angle = np.random.uniform(-2, 2)
            M = cv2.getRotationMatrix2D((base_size // 2, base_size // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (base_size, base_size))

        # Add Gaussian noise
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        images.append(img)

    return images


def demo_basic_stitching():
    """Demonstrate basic panorama stitching."""
    print("Demo: Basic Panorama Stitching")
    print("-" * 50)

    # Generate synthetic images
    print("Generating synthetic images...")
    images = generate_synthetic_images(num_images=3, overlap=0.4)

    # Display input images
    combined_input = np.hstack(images)
    cv2.imshow('Input Images', combined_input)
    cv2.waitKey(1000)

    # Create stitcher
    stitcher = PanoramaStitcher(
        detector='sift',
        matcher='flann',
        blend_mode='linear'
    )

    # Stitch panorama
    print("\nStitching panorama...")
    panorama = stitcher.stitch(images, visualize=True)

    if panorama is not None:
        print("\nPanorama created successfully!")
        cv2.imshow('Panorama', panorama)
        cv2.waitKey(0)
    else:
        print("\nFailed to create panorama")

    cv2.destroyAllWindows()


def demo_comparison():
    """Compare different detector/matcher combinations."""
    print("\nDemo: Detector/Matcher Comparison")
    print("-" * 50)

    # Generate synthetic images
    images = generate_synthetic_images(num_images=2, overlap=0.5)

    configurations = [
        ('sift', 'flann'),
        ('sift', 'bf'),
        ('orb', 'bf'),
    ]

    for detector, matcher in configurations:
        print(f"\n{detector.upper()} + {matcher.upper()}")
        print("-" * 30)

        try:
            stitcher = PanoramaStitcher(detector=detector, matcher=matcher)
            panorama = stitcher.stitch(images.copy())

            if panorama is not None:
                cv2.imshow(f'{detector.upper()}-{matcher.upper()}', panorama)
                cv2.waitKey(1500)
        except Exception as e:
            print(f"Error: {e}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Panorama Stitching Demonstrations")
    print("=" * 50)

    try:
        # Demo 1: Basic stitching
        demo_basic_stitching()

        # Demo 2: Comparison
        demo_comparison()

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    cv2.destroyAllWindows()
    print("\nAll demos completed!")
