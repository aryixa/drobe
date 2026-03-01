# The Wardrobe Vault

A minimalist wardrobe management application built with React, TypeScript, Tailwind CSS, and Supabase.

## Features

- **Authentication**: Secure user authentication with Supabase Auth
- **Wardrobe Management**: Upload, organize, and manage your clothing items
- **Smart Search**: Search your wardrobe with category filters
- **AI Stylist**: Get personalized styling advice and outfit recommendations
- **Responsive Design**: Beautiful minimalist UI that works on all devices
- **Real-time Updates**: Optimistic UI with instant feedback

## Tech Stack

- **Frontend**: React 18, TypeScript, Vite
- **Styling**: Tailwind CSS with custom design system
- **Backend**: Supabase (Database, Auth, Storage)
- **Icons**: Lucide React
- **Routing**: React Router v6

## Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn
- A Supabase account

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd wardrobe-vault
npm install
```

### 2. Supabase Setup

1. Create a new project at [supabase.com](https://supabase.com)
2. Note your `Project URL` and `anon public` key from Settings > API

#### Database Setup

Run this SQL in the Supabase SQL Editor:

```sql
-- Create wardrobe_items table
create table public.wardrobe_items (
  id uuid default gen_random_uuid() primary key,
  user_id uuid references auth.users not null,
  name text not null,
  category text not null default 'top' check (category in ('top', 'bottom', 'shoes', 'accessory')),
  image_url text not null,
  image_path text not null,
  created_at timestamp with time zone default now()
);

-- Performance Indexes
create index on public.wardrobe_items (user_id);
create index on public.wardrobe_items (category);

-- Enable RLS
alter table public.wardrobe_items enable row level security;

-- Policies
create policy "Users can view their own items" on public.wardrobe_items
  for select using (auth.uid() = user_id);

create policy "Users can insert their own items" on public.wardrobe_items
  for insert with check (auth.uid() = user_id);

create policy "Users can delete their own items" on public.wardrobe_items
  for delete using (auth.uid() = user_id);
```

#### Storage Setup

1. Go to Storage in your Supabase dashboard
2. Create a new bucket named `wardrobe` 
3. Make it **Public**
4. Add these policies:

```sql
-- Allow authenticated users to upload
create policy "Authenticated users can upload" on storage.objects
  for insert with check (
    bucket_id = 'wardrobe' and 
    auth.role() = 'authenticated'
  );

-- Allow authenticated users to download
create policy "Authenticated users can download" on storage.objects
  for select using (
    bucket_id = 'wardrobe' and 
    auth.role() = 'authenticated'
  );

-- Allow users to delete their own files
create policy "Users can delete their own files" on storage.objects
  for delete using (
    bucket_id = 'wardrobe' and 
    auth.uid()::text = (storage.foldername(name))[1]
  );
```

### 3. Environment Setup

Create a `.env` file in the root:

```env
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### 4. Run the Application

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
src/
  components/
    ui/           # Reusable UI components
    layout/       # Layout components
    wardrobe/     # Wardrobe-specific components
  context/        # React contexts
  hooks/          # Custom hooks
  lib/            # Utilities and configurations
  pages/          # Page components
```

## Key Features Explained

### Wardrobe Management
- Upload images with validation (JPEG/PNG, max 15MB)
- Organize items by category (tops, bottoms, shoes, accessories)
- Search and filter functionality
- Optimistic UI for instant feedback

### Authentication
- Secure signup/signin with Supabase Auth
- Protected routes
- Automatic session management

### AI Stylist
- Interactive chat interface
- Personalized styling advice
- Quick suggestions for common questions

## Development

### Available Scripts

```bash
npm run dev      # Start development server
npm run build    # Build for production
npm run preview  # Preview production build
npm run lint     # Run ESLint
```

### Design System

The application uses a custom design system built on Tailwind CSS:

- **Colors**: Neutral grayscale palette with accent colors
- **Typography**: Inter font for clean, modern look
- **Components**: Consistent, reusable components with proper TypeScript types
- **Animations**: Subtle transitions and micro-interactions

## Production Deployment

### Build

```bash
npm run build
```

### Environment Variables for Production

Make sure to set these in your hosting environment:
- `VITE_SUPABASE_URL`
- `VITE_SUPABASE_ANON_KEY`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.
