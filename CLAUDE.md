# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Jekyll-based personal CV/resume website for a researcher specializing in Solar Physics and Space Weather. Deployed to www.eunsu.me via GitHub Pages.

## Development Commands

```bash
# Docker (recommended)
docker-compose up
# Access at http://localhost:4000

# Local machine
bundle install
bundle exec jekyll serve
```

## Architecture

**Content Management**: All CV content is centralized in `_data/data.yml`. This single YAML file controls:
- Sidebar (profile info, contact, languages, interests)
- Career profile
- Education, experiences, publications
- Projects, patents

**Layouts**:
- `_layouts/default.html` - Main page structure
- `_layouts/print.html` - Print-optimized view (accessible at `/print`)
- `_layouts/compress.html` - HTML compression wrapper

**Styling**:
- SCSS files in `_sass/`
- 6 color themes in `_sass/skins/` (blue, turquoise, green, berry, orange, ceramic)
- Theme selection via `theme_skin` in `_config.yml`
- Bootstrap 3.x and Font Awesome included in `assets/plugins/`

**Include Components**: Reusable HTML partials in `_includes/` for each CV section (education.html, experiences.html, publications.html, etc.)

## Key Files

- `_data/data.yml` - Edit this for all content changes
- `_config.yml` - Site configuration (theme, URL, analytics)
- `assets/js/main.js` - Skill bar animations (jQuery)
