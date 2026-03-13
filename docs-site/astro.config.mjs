// @ts-check
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

const __dirname = dirname(fileURLToPath(import.meta.url));
const siteConfig = JSON.parse(
	readFileSync(join(__dirname, '../config/site.json'), 'utf-8'),
);
const docsBase = `${siteConfig.base_path}/${siteConfig.docs_subpath}`;

// https://astro.build/config
export default defineConfig({
	site: siteConfig.site_origin,
	server: { host: true },
	preview: { host: true },
	// Use base path for GitHub Pages deployment (docs live at /web-image-formats-research/docs/)
	// For local dev/preview, leave it empty so assets load correctly
	base: process.env.GITHUB_PAGES === 'true' ? docsBase : undefined,
	integrations: [
		starlight({
			title: siteConfig.site_name,
			description: siteConfig.site_description,
			logo: {
				src: './src/assets/logo.svg',
				alt: siteConfig.site_name,
			},
			favicon: '/assets/favicon.ico',
			head: [
				{ tag: 'link', attrs: { rel: 'icon', href: `${docsBase}/assets/favicon.ico`, sizes: '32x32' } },
				{ tag: 'link', attrs: { rel: 'icon', href: `${docsBase}/assets/icon.svg`, type: 'image/svg+xml' } },
				{ tag: 'link', attrs: { rel: 'apple-touch-icon', href: `${docsBase}/assets/apple-touch-icon.png` } },
				{ tag: 'link', attrs: { rel: 'manifest', href: `${docsBase}/assets/manifest.webmanifest` } },
				{ tag: 'meta', attrs: { name: 'theme-color', content: siteConfig.brand.accent } },
				{ tag: 'meta', attrs: { property: 'og:image', content: `${siteConfig.site_origin}${docsBase}/assets/opengraph.png` } },
				{ tag: 'meta', attrs: { name: 'twitter:image', content: `${siteConfig.site_origin}${docsBase}/assets/opengraph.png` } },
			],
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: siteConfig.repository_url,
				},
			],
			sidebar: [
				{
					label: 'Start Here',
					items: [
						{ label: 'Welcome', slug: 'index' },
					],
				},
				{
					label: 'Tutorials',
					collapsed: false,
					autogenerate: { directory: 'tutorials' },
				},
				{
					label: 'How-To Guides',
					collapsed: false,
					autogenerate: { directory: 'how-to' },
				},
				{
					label: 'Reference',
					collapsed: true,
					autogenerate: { directory: 'reference' },
				},
				{
					label: 'Explanation',
					collapsed: true,
					autogenerate: { directory: 'explanation' },
				},
			],
			customCss: [
				'./src/styles/custom.css',
			],
		}),
	],
});
