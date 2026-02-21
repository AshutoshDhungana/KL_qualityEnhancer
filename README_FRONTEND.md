# Knowledge Library Optimization Dashboard

This frontend component provides a modern user interface for the Knowledge Library Optimization system, allowing users to visualize and manage the clustering, similarity detection, and content optimization processes.

## Key Features

- **Cluster Visualization**: View and manage similar content clusters with intuitive interface
- **Content Comparison**: Compare entries side-by-side and see detailed similarity analysis
- **Outdated Content Management**: Easily identify and update outdated knowledge entries
- **Analytics Dashboard**: Get insights into optimization potential and content distribution

## Getting Started

### Prerequisites

- Node.js (version 14 or higher)
- npm (comes with Node.js)

### Installation

1. Navigate to the frontend directory:

```
cd frontend
```

2. Install dependencies:

```
npm install
```

3. Start the development server:

```
npm start
```

The application will be available at http://localhost:3000

### Quick Start

For Windows users, you can also run the `setup.bat` file in the frontend directory to automatically install dependencies and start the application.

## Structure and Components

The application consists of several key components:

1. **Content Clusters** - Displays groups of similar QnA pairs identified by the clustering algorithm
2. **Outdated Content** - Shows content that hasn't been updated in a significant period
3. **Analytics** - Provides visualization of content distribution and optimization metrics

## Integration with Backend

This frontend is designed to work with the clustering and analysis backend. The current implementation uses mock data, but can be connected to the actual API endpoints by modifying the data fetching functions.

## Technologies Used

- React 18
- TypeScript
- Tailwind CSS
- Lucide React icons
