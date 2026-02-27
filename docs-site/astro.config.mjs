// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: 'https://kadykov.github.io',
	server: { host: true },
	preview: { host: true },
	// Use base path for GitHub Pages deployment (docs live at /web-image-formats-research/docs/)
	// For local dev/preview, leave it empty so assets load correctly
	base: process.env.GITHUB_PAGES === 'true' ? '/web-image-formats-research/docs' : undefined,
	integrations: [
		starlight({
			title: 'Web Image Formats Research',
			description: 'Research project for determining optimal modern image formats for the web',
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/kadykov/web-image-formats-research',
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
				// Optional: add custom CSS if needed
			],
		}),
	],
});
