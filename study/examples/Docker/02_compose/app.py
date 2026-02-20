"""
Flask application with PostgreSQL and Redis integration.

This demonstrates:
- Database connections in a containerized environment
- Cache integration for improved performance
- Environment-based configuration
- Health checks for all dependencies
"""

from flask import Flask, jsonify, request
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import os
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================================
# Configuration
# ============================================================================
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/mydb')
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')


# ============================================================================
# Database Connection
# ============================================================================
def get_db_connection():
    """Create a connection to PostgreSQL database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise


def init_db():
    """Initialize database with required tables."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Create a simple visitors table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS visitors (
                id SERIAL PRIMARY KEY,
                ip_address VARCHAR(45),
                user_agent TEXT,
                visited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")


# ============================================================================
# Redis Connection
# ============================================================================
def get_redis_client():
    """Create a Redis client connection."""
    try:
        client = redis.from_url(REDIS_URL, decode_responses=True)
        return client
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        raise


# ============================================================================
# Routes
# ============================================================================
@app.route('/')
def index():
    """Main endpoint with visitor tracking."""
    try:
        # Get visitor info
        ip = request.remote_addr
        user_agent = request.headers.get('User-Agent', 'Unknown')

        # Store in database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO visitors (ip_address, user_agent) VALUES (%s, %s) RETURNING id",
            (ip, user_agent)
        )
        visitor_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        # Increment visitor count in Redis
        r = get_redis_client()
        total_visits = r.incr('total_visits')

        return jsonify({
            'message': 'Welcome to the Flask + PostgreSQL + Redis demo!',
            'visitor_id': visitor_id,
            'total_visits': total_visits,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stats')
def stats():
    """Get visitor statistics."""
    try:
        # Check cache first
        r = get_redis_client()
        cached_stats = r.get('stats_cache')

        if cached_stats:
            logger.info("Returning cached statistics")
            return jsonify(json.loads(cached_stats))

        # Query database
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("SELECT COUNT(*) as total FROM visitors")
        total = cur.fetchone()['total']

        cur.execute("""
            SELECT ip_address, COUNT(*) as visits
            FROM visitors
            GROUP BY ip_address
            ORDER BY visits DESC
            LIMIT 5
        """)
        top_visitors = cur.fetchall()

        cur.close()
        conn.close()

        stats_data = {
            'total_visitors': total,
            'top_visitors': [dict(v) for v in top_visitors],
            'cached': False,
            'timestamp': datetime.now().isoformat()
        }

        # Cache for 60 seconds
        r.setex('stats_cache', 60, json.dumps(stats_data))

        return jsonify(stats_data)

    except Exception as e:
        logger.error(f"Error in stats route: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """
    Comprehensive health check endpoint.

    Checks:
    - Application is running
    - Database connectivity
    - Redis connectivity
    """
    health_status = {
        'status': 'healthy',
        'checks': {}
    }

    # Check database
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT 1')
        cur.close()
        conn.close()
        health_status['checks']['database'] = 'healthy'
    except Exception as e:
        health_status['status'] = 'unhealthy'
        health_status['checks']['database'] = f'unhealthy: {str(e)}'

    # Check Redis
    try:
        r = get_redis_client()
        r.ping()
        health_status['checks']['redis'] = 'healthy'
    except Exception as e:
        health_status['status'] = 'unhealthy'
        health_status['checks']['redis'] = f'unhealthy: {str(e)}'

    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code


@app.route('/cache/clear')
def clear_cache():
    """Clear Redis cache."""
    try:
        r = get_redis_client()
        r.delete('stats_cache')
        return jsonify({'message': 'Cache cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Application Startup
# ============================================================================
if __name__ == '__main__':
    # Initialize database on startup
    try:
        init_db()
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        logger.warning("Continuing anyway - database might not be ready yet")

    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting Flask application on port {port}")

    app.run(host='0.0.0.0', port=port, debug=False)
