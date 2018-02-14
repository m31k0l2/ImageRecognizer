import java.io.File

class ImageNet(populationSize: Int, scale: Int, mutantRate: Double=0.1): AbstractEvolution(populationSize, scale, mutantRate) {
    private val savePerEpoch = 100
    private val folder = "nets/"
    private lateinit var population: List<Individual>

    override fun createNet() = Network(8, 12, 160, 40, 10)

    init {
        if (File("nets/").mkdir()) {
            println("Создаю каталог nets/")
        }
        if (File(folder).mkdir()) {
            println("Создаю каталог $folder")
        }
    }

    // Переопределяем с целью иметь возможность записи и загрузки
    override fun generatePopulation(size: Int): List<Individual> {
        if (!File("$folder/nw.net").exists()) return super.generatePopulation(size)
        val nw = NetworkIO().load("$folder/nw.net")!!
        return super.generatePopulationFrom(Individual(nw), size)
    }

    // переопределяем, чтобы контролировать процесс обучения
    override fun evoluteEpoch(initPopulation: List<Individual>): List<Individual> {
        val start = System.nanoTime()
        println("эпоха $curEpoch")
        population = super.evoluteEpoch(initPopulation)
        if (curEpoch % savePerEpoch == 0) save()
        val fin = System.nanoTime()
        println("Время: ${(fin-start)/1_000_000} мс\n")
        return population
    }

    /** выполняет сохранение сетей на диск **/
    private fun save() {
        println("saving to $folder...")
        NetworkIO().save(population.first().nw, "$folder/nw.net")
    }
}