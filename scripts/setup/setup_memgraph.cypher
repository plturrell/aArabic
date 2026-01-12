// AI Nucleus Platform - Memgraph Sample Data Setup
// This populates the graph database with sample data for GraphChat

// 1. Clear existing data
MATCH (n) DETACH DELETE n;

// 2. Create People nodes
CREATE (alice:Person {name: 'Alice Johnson', email: 'alice@ainucleus.com', role: 'AI Engineer', department: 'Engineering', age: 30})
CREATE (bob:Person {name: 'Bob Smith', email: 'bob@ainucleus.com', role: 'Project Manager', department: 'Management', age: 35})
CREATE (charlie:Person {name: 'Charlie Lee', email: 'charlie@ainucleus.com', role: 'Data Scientist', department: 'Data', age: 28})
CREATE (diana:Person {name: 'Diana Brown', email: 'diana@ainucleus.com', role: 'UX Designer', department: 'Design', age: 32})
CREATE (eve:Person {name: 'Eve Wilson', email: 'eve@ainucleus.com', role: 'DevOps Engineer', department: 'Engineering', age: 29});

// 3. Create Project nodes
CREATE (p1:Project {name: 'AI Nucleus Platform', status: 'Active', priority: 'High', budget: 500000, startDate: '2024-01-01'})
CREATE (p2:Project {name: 'Arabic NLP Pipeline', status: 'Active', priority: 'High', budget: 300000, startDate: '2024-02-01'})
CREATE (p3:Project {name: 'Graph Analytics Engine', status: 'Planning', priority: 'Medium', budget: 200000, startDate: '2024-03-01'})
CREATE (p4:Project {name: 'Workflow Automation', status: 'Active', priority: 'Medium', budget: 150000, startDate: '2024-01-15'});

// 4. Create Technology nodes
CREATE (t1:Technology {name: 'Python', type: 'Language', version: '3.11'})
CREATE (t2:Technology {name: 'Memgraph', type: 'Database', version: '2.15.0'})
CREATE (t3:Technology {name: 'N8N', type: 'Automation', version: '1.0'})
CREATE (t4:Technology {name: 'LangChain', type: 'Framework', version: '0.1'})
CREATE (t5:Technology {name: 'FastAPI', type: 'Framework', version: '0.109'});

// 5. Create Department nodes
CREATE (eng:Department {name: 'Engineering', headcount: 15, budget: 2000000})
CREATE (data:Department {name: 'Data', headcount: 8, budget: 1000000})
CREATE (design:Department {name: 'Design', headcount: 5, budget: 500000})
CREATE (mgmt:Department {name: 'Management', headcount: 3, budget: 800000});

// 6. Create relationships - Work assignments
MATCH (alice:Person {name: 'Alice Johnson'}), (p1:Project {name: 'AI Nucleus Platform'})
CREATE (alice)-[:WORKS_ON {since: '2024-01-01', hoursPerWeek: 40, role: 'Lead Engineer'}]->(p1);

MATCH (alice:Person {name: 'Alice Johnson'}), (p2:Project {name: 'Arabic NLP Pipeline'})
CREATE (alice)-[:WORKS_ON {since: '2024-02-01', hoursPerWeek: 20, role: 'Technical Advisor'}]->(p2);

MATCH (bob:Person {name: 'Bob Smith'}), (p1:Project {name: 'AI Nucleus Platform'})
CREATE (bob)-[:MANAGES {since: '2024-01-01'}]->(p1);

MATCH (bob:Person {name: 'Bob Smith'}), (p4:Project {name: 'Workflow Automation'})
CREATE (bob)-[:MANAGES {since: '2024-01-15'}]->(p4);

MATCH (charlie:Person {name: 'Charlie Lee'}), (p2:Project {name: 'Arabic NLP Pipeline'})
CREATE (charlie)-[:WORKS_ON {since: '2024-02-01', hoursPerWeek: 40, role: 'Data Scientist'}]->(p2);

MATCH (charlie:Person {name: 'Charlie Lee'}), (p3:Project {name: 'Graph Analytics Engine'})
CREATE (charlie)-[:WORKS_ON {since: '2024-03-01', hoursPerWeek: 10, role: 'Analyst'}]->(p3);

MATCH (diana:Person {name: 'Diana Brown'}), (p1:Project {name: 'AI Nucleus Platform'})
CREATE (diana)-[:WORKS_ON {since: '2024-01-01', hoursPerWeek: 30, role: 'UX Designer'}]->(p1);

MATCH (eve:Person {name: 'Eve Wilson'}), (p1:Project {name: 'AI Nucleus Platform'})
CREATE (eve)-[:WORKS_ON {since: '2024-01-01', hoursPerWeek: 40, role: 'DevOps Engineer'}]->(p1);

// 7. Create colleague relationships
MATCH (alice:Person {name: 'Alice Johnson'}), (bob:Person {name: 'Bob Smith'})
CREATE (alice)-[:COLLEAGUE_OF {since: '2024-01-01'}]->(bob);

MATCH (alice:Person {name: 'Alice Johnson'}), (charlie:Person {name: 'Charlie Lee'})
CREATE (alice)-[:COLLEAGUE_OF {since: '2024-02-01'}]->(charlie);

MATCH (alice:Person {name: 'Alice Johnson'}), (diana:Person {name: 'Diana Brown'})
CREATE (alice)-[:COLLEAGUE_OF {since: '2024-01-01'}]->(diana);

MATCH (alice:Person {name: 'Alice Johnson'}), (eve:Person {name: 'Eve Wilson'})
CREATE (alice)-[:COLLEAGUE_OF {since: '2024-01-01'}]->(eve);

MATCH (bob:Person {name: 'Bob Smith'}), (charlie:Person {name: 'Charlie Lee'})
CREATE (bob)-[:COLLEAGUE_OF {since: '2024-02-01'}]->(charlie);

// 8. Create department memberships
MATCH (alice:Person {name: 'Alice Johnson'}), (eng:Department {name: 'Engineering'})
CREATE (alice)-[:BELONGS_TO]->(eng);

MATCH (bob:Person {name: 'Bob Smith'}), (mgmt:Department {name: 'Management'})
CREATE (bob)-[:BELONGS_TO]->(mgmt);

MATCH (charlie:Person {name: 'Charlie Lee'}), (data:Department {name: 'Data'})
CREATE (charlie)-[:BELONGS_TO]->(data);

MATCH (diana:Person {name: 'Diana Brown'}), (design:Department {name: 'Design'})
CREATE (diana)-[:BELONGS_TO]->(design);

MATCH (eve:Person {name: 'Eve Wilson'}), (eng:Department {name: 'Engineering'})
CREATE (eve)-[:BELONGS_TO]->(eng);

// 9. Create technology usage
MATCH (p1:Project {name: 'AI Nucleus Platform'}), (t1:Technology {name: 'Python'})
CREATE (p1)-[:USES {since: '2024-01-01'}]->(t1);

MATCH (p1:Project {name: 'AI Nucleus Platform'}), (t2:Technology {name: 'Memgraph'})
CREATE (p1)-[:USES {since: '2024-01-01'}]->(t2);

MATCH (p1:Project {name: 'AI Nucleus Platform'}), (t5:Technology {name: 'FastAPI'})
CREATE (p1)-[:USES {since: '2024-01-01'}]->(t5);

MATCH (p2:Project {name: 'Arabic NLP Pipeline'}), (t1:Technology {name: 'Python'})
CREATE (p2)-[:USES {since: '2024-02-01'}]->(t1);

MATCH (p2:Project {name: 'Arabic NLP Pipeline'}), (t4:Technology {name: 'LangChain'})
CREATE (p2)-[:USES {since: '2024-02-01'}]->(t4);

MATCH (p4:Project {name: 'Workflow Automation'}), (t3:Technology {name: 'N8N'})
CREATE (p4)-[:USES {since: '2024-01-15'}]->(t3);

// 10. Verify data was created
MATCH (n) RETURN labels(n)[0] as NodeType, count(n) as Count ORDER BY NodeType;
