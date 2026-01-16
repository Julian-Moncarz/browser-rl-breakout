/**
 * Headless Breakout environment for RL training.
 * Extracted from game.html with minimal modifications.
 */

export function createEnv(config = {}) {
    const width = config.width || 800;
    const height = config.height || 600;

    // Paddle
    const paddle = {
        width: 120,
        height: 15,
        x: 340,
        y: 560,
        speed: 8
    };

    // Ball
    const ball = {
        x: 400,
        y: 545,
        radius: 8,
        dx: 5,
        dy: -5,
        speed: 5
    };

    // Brick config
    const brickRows = 5;
    const brickCols = 10;
    const brickWidth = 70;
    const brickHeight = 25;
    const brickPadding = 8;
    const brickOffsetTop = 60;
    const brickOffsetLeft = 35;

    // Game state
    let bricks = [];
    let ballLaunched = false;
    let score = 0;
    let prevScore = 0;
    let lives = 3;
    let level = 1;
    let combo = 0;

    function createBricks() {
        bricks = [];
        for (let row = 0; row < brickRows; row++) {
            for (let col = 0; col < brickCols; col++) {
                const hits = row < 2 ? 2 : 1;
                bricks.push({
                    x: brickOffsetLeft + col * (brickWidth + brickPadding),
                    y: brickOffsetTop + row * (brickHeight + brickPadding),
                    width: brickWidth,
                    height: brickHeight,
                    hits: hits,
                    maxHits: hits,
                    alive: true
                });
            }
        }
    }

    function resetBall() {
        ball.x = paddle.x + paddle.width / 2;
        ball.y = paddle.y - ball.radius - 2;
        ball.dx = (Math.random() - 0.5) * 6;
        ball.dy = -ball.speed;
        ballLaunched = false;
    }

    function reset() {
        score = 0;
        prevScore = 0;
        lives = 3;
        level = 1;
        combo = 0;
        ball.speed = 5;
        paddle.x = 340;
        createBricks();
        resetBall();
        return getObs();
    }

    function launch() {
        if (!ballLaunched) {
            ballLaunched = true;
            ball.dy = -ball.speed;
            ball.dx = (Math.random() - 0.5) * 6;
        }
    }

    function getObs() {
        // Count bricks alive per row
        const rowCounts = new Array(brickRows).fill(0);
        bricks.forEach((b, i) => {
            if (b.alive) rowCounts[Math.floor(i / brickCols)]++;
        });

        return new Float32Array([
            ball.x / width,           // 0: ball x normalized
            ball.y / height,          // 1: ball y normalized
            ball.dx / 10,             // 2: ball dx (small range)
            ball.dy / 10,             // 3: ball dy (small range)
            paddle.x / width,         // 4: paddle x normalized
            ballLaunched ? 1 : 0,     // 5: ball launched flag
            lives / 3,                // 6: lives normalized
            Math.min(level, 10) / 10, // 7: level normalized (cap at 10)
            ...rowCounts.map(c => c / brickCols) // 8-12: bricks per row normalized
        ]);
    }

    function applyAction(action) {
        if (action === 0) paddle.x -= paddle.speed;
        else if (action === 2) paddle.x += paddle.speed;
        paddle.x = Math.max(0, Math.min(width - paddle.width, paddle.x));
    }

    function step(action) {
        prevScore = score;
        let lifeLost = false;
        let levelCleared = false;

        // Apply action
        applyAction(action);

        if (!ballLaunched) {
            ball.x = paddle.x + paddle.width / 2;
            ball.y = paddle.y - ball.radius - 2;
            return { obs: getObs(), reward: 0, done: false, info: { lifeLost, levelCleared } };
        }

        // Ball movement
        ball.x += ball.dx;
        ball.y += ball.dy;

        // Wall collision
        if (ball.x <= ball.radius || ball.x >= width - ball.radius) {
            ball.dx *= -1;
            ball.x = Math.max(ball.radius, Math.min(width - ball.radius, ball.x));
        }
        if (ball.y <= ball.radius) {
            ball.dy *= -1;
            ball.y = ball.radius;
        }

        // Bottom - lose life
        if (ball.y >= height + ball.radius) {
            lives--;
            combo = 0;
            lifeLost = true;
            if (lives <= 0) {
                return { 
                    obs: getObs(), 
                    reward: -100, 
                    done: true, 
                    info: { lifeLost, levelCleared } 
                };
            } else {
                resetBall();
                ballLaunched = true; // Auto-relaunch for training
                ball.dy = -ball.speed;
                ball.dx = (Math.random() - 0.5) * 6;
            }
        }

        // Paddle collision
        if (ball.y + ball.radius >= paddle.y &&
            ball.y - ball.radius <= paddle.y + paddle.height &&
            ball.x >= paddle.x &&
            ball.x <= paddle.x + paddle.width) {
            
            const hitPoint = (ball.x - paddle.x) / paddle.width;
            ball.dx = (hitPoint - 0.5) * 10;
            ball.dy = -Math.abs(ball.dy);
            ball.y = paddle.y - ball.radius;
            combo = 0;
        }

        // Brick collision
        bricks.forEach(brick => {
            if (!brick.alive) return;

            if (ball.x + ball.radius > brick.x &&
                ball.x - ball.radius < brick.x + brick.width &&
                ball.y + ball.radius > brick.y &&
                ball.y - ball.radius < brick.y + brick.height) {
                
                brick.hits--;
                if (brick.hits <= 0) {
                    brick.alive = false;
                    combo++;
                    const points = 10 * combo * level;
                    score += points;
                }

                // Determine bounce direction
                const overlapLeft = ball.x + ball.radius - brick.x;
                const overlapRight = brick.x + brick.width - (ball.x - ball.radius);
                const overlapTop = ball.y + ball.radius - brick.y;
                const overlapBottom = brick.y + brick.height - (ball.y - ball.radius);

                const minOverlapX = Math.min(overlapLeft, overlapRight);
                const minOverlapY = Math.min(overlapTop, overlapBottom);

                if (minOverlapX < minOverlapY) {
                    ball.dx *= -1;
                } else {
                    ball.dy *= -1;
                }
            }
        });

        // Check level complete
        if (bricks.every(b => !b.alive)) {
            level++;
            ball.speed += 0.5;
            levelCleared = true;
            createBricks();
            resetBall();
            ballLaunched = true; // Auto-relaunch for training
            ball.dy = -ball.speed;
            ball.dx = (Math.random() - 0.5) * 6;
        }

        // Calculate reward
        let reward = score - prevScore; // Points from breaking bricks
        if (lifeLost) reward -= 100;
        if (levelCleared) reward += 500;

        return { 
            obs: getObs(), 
            reward, 
            done: false, 
            info: { lifeLost, levelCleared, score, lives, level } 
        };
    }

    return {
        reset,
        step,
        getObs,
        launch,
        // Expose for debugging
        getState: () => ({ ball: { ...ball }, paddle: { ...paddle }, lives, score, level, ballLaunched })
    };
}
