/**
 * SVGO configuration for web report SVG optimisation.
 *
 * Extends the default preset with ``removeStyleElement``, which strips
 * embedded ``<style>`` blocks from Matplotlib-generated SVGs.  Those
 * blocks encode font and colour rules that are already inlined on the
 * individual elements, so removing them yields a significant size
 * reduction with no visual change.
 *
 * ``removeStyleElement`` is disabled in SVGO's built-in preset because
 * it is unsafe for arbitrary SVGs, but it is safe for our Matplotlib
 * output where all styling is redundantly duplicated as element attrs.
 *
 * Numeric precision is controlled separately via the ``--precision``
 * CLI flag passed by :func:`src.report_images.optimise_svg`.
 */
export default {
  plugins: [
    {
      name: "preset-default",
      params: {
        overrides: {},
      },
    },
    "removeStyleElement",
  ],
};
