import psycopg2

# Establish connection to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="secPost#1"
)

# Create a cursor object
cur = conn.cursor()

# Write a SQL query
query = "SELECT * FROM public.xdr_data LIMIT 10;"

# Execute the query
cur.execute(query)

# Fetch the results
rows = cur.fetchall()

# Print the results
for row in rows:
    print(row)

# Close the cursor and connection
cur.close()
conn.close()
