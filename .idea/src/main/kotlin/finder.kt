import java.util.*

class Tree {
    private val state = mutableMapOf<String, Node>()
    private val distanceMap = mutableMapOf<String, Int>()

    class Node(val id: String) {
        val children = mutableSetOf<Node>()

        fun add(id: String): Node {
            val next = Node(id)
            next.children.add(this)
            children.add(next)
            return next
        }
    }

    fun add(from: String, to: String, distance: Int=0) {
        if (state.isEmpty()) state[from] = Node(from)
        val parent = state[from]
        if (parent != null) {
            val children = parent.add(to)
            state[to] = children
            distanceMap["$from-$to"] = distance
        } else {
            add(to, from, distance)
        }
    }

    fun distanceBetween(from: String, to: String) = distanceMap["$from-$to"] ?: distanceMap["$to-$from"]

    fun get(id: String) = state[id]
}

val map = Tree()

fun main(args: Array<String>) {
    map.add("Arad", "Zerind", 75)
    map.add("Arad", "Timisoara", 118)
    map.add("Arad", "Sibiu", 140)
    map.add("Zerind", "Oradea", 71)
    map.add("Oradea", "Sibiu", 151)
    map.add("Timisoara", "Lugoj", 111)
    map.add("Lugoj", "Mehadia", 70)
    map.add("Mehadia", "Drobeta", 75)
    map.add("Drobeta", "Craiova", 120)
    map.add("Craiova", "Rimnicu Vilcea", 146)
    map.add("Sibiu", "Rimnicu Vilcea", 80)
    map.add("Sibiu", "Fagaras", 99)
    map.add("Fagaras", "Bucharest", 211)
    map.add("Bucharest", "Pitesti", 101)
    map.add("Bucharest", "Giurgiu", 90)
    map.add("Bucharest", "Urziceni", 85)
    map.add("Urziceni", "Vaslui", 142)
    map.add("Vaslui", "Iasi", 92)
    map.add("Neamt", "Iasi", 87)
    map.add("Hirsova", "Urziceni", 98)
    map.add("Hirsova", "Eforie", 86)
    map.add("Craiova", "Pitesti", 138)
    println(map.distanceBetween("Arad", "Timisoara"))

    val node = map.get("Arad")!!
    val next = node.children.minBy { map.distanceBetween(node.id, it.id)!! }!!
    println(next.id)
    map.treeSearch("Arad", "Bucharest")
}

class TreeSearchException(msg: String): Exception(msg)

fun Node.isGoal(node: Tree.Node) = node.id == state.id

data class Node(var state: Tree.Node, var parentNode: Node?, var pathCost: Int, var depth: Int) {
    var action = emptyList<Node>()
}

fun makeNode(state: Tree.Node, parentAction: List<Node>): Node {
    if (parentAction.isEmpty()) {
        return Node(state, null, 0, 1)
    }
    val parentNode = parentAction.last()
    val cost = map.distanceBetween(parentNode.state.id, state.id)!!
    val node = Node(state, parentNode, parentNode.pathCost + cost, parentNode.depth + 1)
    node.action = listOf(*parentAction.toTypedArray(), node)
    return node
}

fun Node.expand() = state.children.map { makeNode(it, action) }

fun Tree.treeSearch(root: String, problem: String): List<Tree.Node> {
    val start = get(root) ?: throw TreeSearchException("Не найден пункт $root")
    val finish = get(problem) ?: throw TreeSearchException("Не найден пункт $problem")
    val fringe = LinkedList<Node>()//mutableListOf(makeNode(start, emptyList()))
    fringe.add(makeNode(start, emptyList()))
    val solution = mutableListOf(start)
    while (true) {
        if (fringe.isEmpty()) throw TreeSearchException("Нет кандидатов на развёртывание")
        val node = fringe.poll()
        if (node.isGoal(finish)) return solution
        fringe.addAll(node.expand())
    }
}