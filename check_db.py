import sqlite3

# Connect to database and check contents
conn = sqlite3.connect('data/raw/patches.db')
cursor = conn.cursor()

# Check table schema
cursor.execute("PRAGMA table_info(patterns)")
print("Table schema:")
for row in cursor.fetchall():
    print(f"  Column: {row[1]}, Type: {row[2]}, NotNull: {row[3]}, Default: {row[4]}, PrimaryKey: {row[5]}")

print()

# Get total count
cursor.execute('SELECT COUNT(*) FROM patterns')
total = cursor.fetchone()[0]
print(f'Total patterns: {total}')

# Get count by quality
cursor.execute('SELECT quality, COUNT(*) FROM patterns GROUP BY quality')
print('\nPatterns by quality:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]}')

# Get some sample patterns (complete)
cursor.execute('SELECT edgebreaker_encoding, sides, quality, source_obj FROM patterns LIMIT 3')
print('\nSample patterns (complete):')
for row in cursor.fetchall():
    print(f'  Pattern: {row[0]}')
    print(f'  Sides: {row[1]}')
    print(f'  Quality: {row[2]}')
    print(f'  Source: {row[3]}')
    print('  ---')

# Get patterns from different quality and sides combinations
print('\nSamples from different categories:')
cursor.execute('SELECT edgebreaker_encoding, sides, quality, source_obj FROM patterns WHERE quality="old" LIMIT 2')
print('Old quality patterns:')
for row in cursor.fetchall():
    print(f'  {row[0]} | {row[1]} sides | {row[3]}')

cursor.execute('SELECT edgebreaker_encoding, sides, quality, source_obj FROM patterns WHERE sides=13 LIMIT 2')
print('\n13-sided patterns:')
for row in cursor.fetchall():
    print(f'  {row[0]} | {row[2]} quality | {row[3]}')

cursor.execute('SELECT edgebreaker_encoding, sides, quality, source_obj FROM patterns WHERE sides=19 LIMIT 2')
print('\n19-sided patterns:')
for row in cursor.fetchall():
    print(f'  {row[0]} | {row[2]} quality | {row[3]}')

# Check pattern length distribution
cursor.execute('SELECT MIN(LENGTH(edgebreaker_encoding)), MAX(LENGTH(edgebreaker_encoding)), AVG(LENGTH(edgebreaker_encoding)) FROM patterns')
lengths = cursor.fetchone()
print(f'\nPattern length stats:')
print(f'  Min: {lengths[0]} characters')
print(f'  Max: {lengths[1]} characters')
print(f'  Average: {lengths[2]:.1f} characters')

# Get count by sides
cursor.execute('SELECT sides, COUNT(*) FROM patterns GROUP BY sides ORDER BY sides')
print('\nPatterns by number of sides:')
for row in cursor.fetchall():
    print(f'  {row[0]} sides: {row[1]} patterns')

# Check unique source objects
cursor.execute('SELECT DISTINCT source_obj FROM patterns')
print('\nSource objects:')
for row in cursor.fetchall():
    print(f'  {row[0]}')

conn.close() 